#!/usr/bin/env python3
"""
Merge LoRA adapter into Qwen3.5-9B base, saving an FP16 HuggingFace folder
ready for llama.cpp's convert_hf_to_gguf.py.

Why this exists:
    Unsloth's `save_pretrained_merged` has a documented bug for vision models
    (https://github.com/unslothai/unsloth/issues/1352) — adapters are silently
    NOT merged. We use peft's `merge_and_unload()` directly, which has been
    verified to work for Qwen vision models.

CRITICAL: Gated DeltaNet rogue-target stripping
    The training-time LoRA `target_modules` regex (Unsloth default) contains
    substrings `qkv` and `in_proj_b`, which UNINTENTIONALLY substring-match
    Qwen3.5's fused Gated DeltaNet projections `linear_attn.in_proj_qkvz` and
    `linear_attn.in_proj_ba`. PEFT then attaches LoRA to these fused tensors
    treating them as flat nn.Linear, with no awareness of their internal
    Q/K/V/Z partition.

    `convert_hf_to_gguf.py:_reorder_v_heads()` later reorders the V-slice rows
    of `in_proj_qkvz` from grouped-per-K-head layout to tiled layout. Any
    LoRA delta in the V-slice rows ends up in scrambled head positions →
    degenerate single-token output ("adgeadgeadge...") in 75% of layers
    (the linear-attention blocks).

    Fix: zero `lora_B` for any module whose name contains the rogue suffixes
    BEFORE calling `merge_and_unload()`. This drops the unintended LoRA
    contribution to the linear-attention blocks while preserving the
    correctly-placed LoRA on standard q/k/v/o/gate/up/down projections (the
    25% of layers that are full attention). Output becomes coherent at the
    cost of any linear-attention adaptation the LoRA had learned (which was
    accidental anyway).

    Refs:
      - github.com/ggml-org/llama.cpp/issues/21125 (upstream V-reorder bug)
      - swift `--linear_decoupled_in_proj true` (the proper training fix)

GPU mode:
    Loads base in FP16 directly to GPU (`device_map="auto"`). Tested target:
    RTX A5000 (24 GB VRAM). The 9B FP16 base is ~18 GB, leaving ~6 GB for
    the per-layer merge working set. We move the merged model to CPU
    BEFORE save to free VRAM during shard writing — sharded safetensors
    writes can transiently double tensor memory.

    If you see OOM during merge, set FORCE_CPU=1 to fall back to CPU mode
    (slower but uses system RAM instead of VRAM).

Env:
    HF_TOKEN          (required) — for downloading the private adapter
    BASE_MODEL        default: unsloth/Qwen3.5-9B
    ADAPTER_REPO      default: pfox1995/pest-detector-final
    ADAPTER_LOCAL     optional — local adapter folder; overrides ADAPTER_REPO
    OUTPUT_DIR        default: /workspace/merged_fp16
    FORCE_CPU         optional — set to "1" to disable GPU and use CPU merge

Usage:
    python merge_lora.py
"""

import os
import sys
import shutil
from pathlib import Path

import torch
from huggingface_hub import login, snapshot_download


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not set (needed for private adapter).")

    base_id      = os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-9B")
    adapter_repo = os.environ.get("ADAPTER_REPO", "pfox1995/pest-detector-final")
    adapter_loc  = os.environ.get("ADAPTER_LOCAL", "")
    output_dir   = os.environ.get("OUTPUT_DIR", "/workspace/merged_fp16")
    force_cpu    = os.environ.get("FORCE_CPU", "0") == "1"

    use_gpu = torch.cuda.is_available() and not force_cpu
    device_map = "auto" if use_gpu else "cpu"
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {gpu_name} ({gpu_vram_gb:.1f} GB VRAM)")
        if gpu_vram_gb < 22:
            print(f"WARNING: <22 GB VRAM may OOM on 9B FP16 merge. Set FORCE_CPU=1 if it fails.")
    else:
        print(f"GPU mode {'disabled (FORCE_CPU=1)' if force_cpu else 'unavailable'}; using CPU.")

    login(token=token, add_to_git_credential=False)

    print(f"[1/4] Resolving adapter location...")
    if adapter_loc and Path(adapter_loc).is_dir():
        adapter_path = adapter_loc
        print(f"      Using local adapter: {adapter_path}")
    else:
        print(f"      Downloading adapter: {adapter_repo}")
        adapter_path = snapshot_download(repo_id=adapter_repo, token=token)
        print(f"      Cached at: {adapter_path}")

    print(f"[2/4] Loading base model in FP16 to {device_map}: {base_id}")
    print(f"      (NOT 4-bit — we need full precision weights to merge into)")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = AutoModelForImageTextToText.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(base_id)
    if use_gpu:
        used_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"      Base model loaded. VRAM used: {used_gb:.1f} GB")

    print(f"[3/4] Attaching LoRA and merging into base weights...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        torch_dtype=torch.float16,  # prevent FP32 upcast of adapter weights
    )

    # See module docstring: zero out lora_B on the Gated DeltaNet linear-attn
    # projections before merge. The training-time `target_modules` regex
    # accidentally matched these — Unsloth defaults to decoupled mode where
    # the fused QKVZ/BA tensors are already split into 4 separate Linear
    # modules per layer.
    #
    # SKIP_ZEROING=1 disables this: produces a "complete" merge with all 248
    # LoRA modules baked in. Use this when applying the pre-permute trick
    # (gguf_export/pre_permute_for_gguf.py) so that GGUF conversion's
    # _reorder_v_heads permutation is undone before inference.
    skip_zeroing = os.environ.get("SKIP_ZEROING", "0") == "1"
    rogue_substrings = () if skip_zeroing else (
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        "linear_attn.in_proj_a", "linear_attn.in_proj_b",
        "linear_attn.out_proj",
    )
    if skip_zeroing:
        print(f"      SKIP_ZEROING=1: NOT zeroing rogue projections — full LoRA merge.")
    zeroed = 0
    for name, module in model.named_modules():
        if any(s in name for s in rogue_substrings) and hasattr(module, "lora_B"):
            for adapter_name in module.lora_B:
                with torch.no_grad():
                    module.lora_B[adapter_name].weight.zero_()
                zeroed += 1
    print(f"      Zeroed lora_B on {zeroed} Gated DeltaNet fused projections "
          f"(prevents V-head scrambling at GGUF conversion time).")

    merged = model.merge_and_unload()  # <- the critical call (NOT save_pretrained_merged)
    merged = merged.to(torch.float16)  # belt-and-suspenders: re-assert FP16 post-merge
    print(f"      Merge complete. LoRA layers now baked into base weights.")
    if use_gpu:
        used_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"      Post-merge VRAM: {used_gb:.1f} GB")

    if use_gpu:
        print(f"      Moving merged model to CPU before save (frees VRAM during shard write)...")
        merged = merged.to("cpu")
        torch.cuda.empty_cache()

    print(f"[4/4] Saving FP16 model to: {output_dir}")
    out = Path(output_dir)
    if out.exists():
        print(f"      Cleaning existing dir...")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    processor.save_pretrained(output_dir)

    total_gb = sum(
        f.stat().st_size for f in out.rglob("*") if f.is_file()
    ) / 1024**3
    print(f"\n[done] Wrote {total_gb:.1f} GB to {output_dir}")
    print(f"       Next: python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile model-F16.gguf --outtype f16")


if __name__ == "__main__":
    main()
