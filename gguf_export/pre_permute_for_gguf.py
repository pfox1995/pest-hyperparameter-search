#!/usr/bin/env python3
"""
Pre-permute the merged Qwen3.5-9B FP16 model so that
`convert_hf_to_gguf.py:_reorder_v_heads` is exactly undone at conversion time.

Background
----------
For Qwen3.5 hybrid Gated DeltaNet, llama.cpp's GGUF conversion permutes V-head
rows from the HF "grouped" layout to a "tiled" layout that ggml_repeat can
broadcast over efficiently:
    grouped: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...]   (PyTorch / HF storage)
    tiled:   [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]  (ggml runtime)

For the BASE model this is fine — both layouts are mathematically equivalent
under different binary-op patterns. The CUDA kernel reads tiled.

For a LoRA-merged tensor, the delta `(B @ A) * scaling` was trained on the
PyTorch model where the runtime layout was grouped. After PEFT's
`merge_and_unload()`, the merged tensor is `base_grouped + delta_grouped`.
GGUF conversion then runs `_reorder_v_heads` on the merged tensor, sending
both halves to tiled layout — equivalent to the original training math, BUT
the LoRA delta was trained respecting an internal V-head structure that
the conversion's row-permutation breaks (the delta's V-rows aren't aware
of the K-grouping).

By applying the INVERSE of `_reorder_v_heads` to the merged tensor BEFORE
running `convert_hf_to_gguf.py`, the conversion's forward permutation cancels
our inverse, leaving the GGUF tensor in HF grouped layout. llama.cpp can
still consume this — it just falls back to the slower interleaved-repeat
path instead of the optimized tiled-broadcast.

`_reorder_v_heads` is its own inverse when num_k_heads and num_v_per_k are
swapped (it's a swap of two factor groups in a reshape).

Tensors that get reordered by Qwen3_5TextModel (lines ~5337-5424 of
convert_hf_to_gguf.py):
    - linear_attn.in_proj_qkv.weight  : V-slice rows only (offset q_dim+k_dim)
    - linear_attn.in_proj_z.weight    : entire dim 0
    - linear_attn.in_proj_a.weight    : entire dim 0, head_dim=1
    - linear_attn.in_proj_b.weight    : entire dim 0, head_dim=1
    - linear_attn.out_proj.weight     : entire dim 1 (column reorder)

Usage
-----
    python pre_permute_for_gguf.py /workspace/merged_fp16
        # rewrites safetensors files in-place

The script processes one shard at a time to keep peak memory low.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Tensors to process (matches Qwen3_5TextModel.modify_tensors logic)
PATTERNS = (
    ("in_proj_qkv", 0, "qkv"),       # V-slice rows only
    ("in_proj_z",   0, "full"),
    ("in_proj_a",   0, "full_h1"),   # head_dim=1
    ("in_proj_b",   0, "full_h1"),   # head_dim=1
    ("out_proj",    1, "full"),
)


def reorder_swap(t: torch.Tensor, dim: int, n_outer: int, n_inner: int, head_dim: int) -> torch.Tensor:
    """Apply the SAME swap-reshape used by _reorder_v_heads.

    With (n_outer, n_inner) = (num_k_heads, num_v_per_k, head_dim) → forward.
    With (n_outer, n_inner) = (num_v_per_k, num_k_heads, head_dim) → inverse.
    """
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [n_outer, n_inner, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def process_tensor(name: str, t: torch.Tensor, params: dict) -> torch.Tensor:
    nk = params["num_k_heads"]
    nv_per_k = params["num_v_per_k"]
    hv = params["head_v_dim"]
    q_dim = params["q_dim"]
    k_dim = params["k_dim"]

    if "linear_attn.in_proj_qkv.weight" in name or "linear_attn.in_proj_qkv." in name:
        v_offset = q_dim + k_dim
        v = t[v_offset:]
        # inverse permutation: swap nv_per_k and nk roles
        v_inv = reorder_swap(v, 0, nv_per_k, nk, hv)
        return torch.cat([t[:v_offset], v_inv], dim=0).to(t.dtype)

    if "linear_attn.in_proj_z" in name:
        return reorder_swap(t, 0, nv_per_k, nk, hv).to(t.dtype)

    if "linear_attn.in_proj_a" in name or "linear_attn.in_proj_b" in name:
        return reorder_swap(t, 0, nv_per_k, nk, 1).to(t.dtype)

    if "linear_attn.out_proj" in name:
        return reorder_swap(t, 1, nv_per_k, nk, hv).to(t.dtype)

    return t


def derive_params(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    text_cfg = cfg.get("text_config", cfg)  # vision-language wraps text_config
    nk = text_cfg["linear_num_key_heads"]
    nv = text_cfg["linear_num_value_heads"]
    hk = text_cfg["linear_key_head_dim"]
    hv = text_cfg["linear_value_head_dim"]
    return {
        "num_k_heads": nk,
        "num_v_per_k": nv // nk,
        "head_k_dim": hk,
        "head_v_dim": hv,
        "q_dim": nk * hk,
        "k_dim": nk * hk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_dir", help="merged FP16 model directory")
    ap.add_argument("--dry-run", action="store_true", help="don't save, just count tensors")
    args = ap.parse_args()

    merged = Path(args.merged_dir)
    cfg_path = merged / "config.json"
    if not cfg_path.exists():
        sys.exit(f"missing {cfg_path}")
    params = derive_params(cfg_path)
    print(f"Linear-attn params: {params}")

    shards = sorted(merged.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no .safetensors in {merged}")

    total_processed = 0
    for shard in shards:
        print(f"\n[{shard.name}] loading...", flush=True)
        state = load_file(str(shard))
        local_processed = 0
        for name in list(state.keys()):
            if any(p[0] in name for p in PATTERNS):
                state[name] = process_tensor(name, state[name], params)
                local_processed += 1
        print(f"  permuted {local_processed} tensor(s)")
        total_processed += local_processed
        if not args.dry_run:
            print(f"  saving...", flush=True)
            save_file(state, str(shard), metadata={"format": "pt"})

    print(f"\n[done] permuted {total_processed} tensors across {len(shards)} shards")


if __name__ == "__main__":
    main()
