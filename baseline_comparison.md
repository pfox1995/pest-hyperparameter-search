## 📈 Baseline Comparison: With vs Without the LoRA Adapter

To quantify what the LoRA fine-tuning actually teaches the model, we ran the **identical evaluation pipeline** on the raw `unsloth/Qwen3.5-9B` base model **without any adapter attached** — same 1,595-sample validation split of [`Himedia-AI-01/pest-detection-korean`](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean), same preprocessing (letterbox 512×512), same system prompt, same decoding settings.

Any prediction that does not match one of the 19 valid class labels is counted as **UNKNOWN** (off-vocabulary) and scored as a miss for the true class. This is the honest evaluation of *"does the model produce a correct pest label?"*

### Headline Results

| Metric | Base Qwen3.5-9B<br>(no LoRA) | **With LoRA<br>(pest-detector-final)** | Δ |
|:---|:---:|:---:|:---:|
| **Accuracy** | 9.47% | **89.47%** | **+80.00 pp** |
| **F1 (macro)** | 4.32% | **89.44%** | **+85.12 pp** |
| **F1 (weighted)** | 7.37% | **90.38%** | +83.01 pp |
| Precision (macro) | 8.00% | 90.88% | +82.88 pp |
| Recall (macro) | 5.47% | 89.28% | +83.81 pp |
| **Off-vocabulary outputs** | **1,183 / 1,595 (74.2%)** | **33 / 1,595 (2.1%)** | **−72.1 pp** |

### Confusion Matrices

**Base model (no LoRA)** — almost every row dumps into the right-most `UNKNOWN` column:

![Baseline CM](confusion_matrix_baseline_final.png)

**With LoRA adapter** — predictions concentrate along the diagonal, `UNKNOWN` column near-empty:

![LoRA CM](confusion_matrix_final_with_unknown.png)

---

### What the base model outputs — a closer look

When you examine the 1,183 off-vocabulary predictions from the base model, something surprising emerges: **they are not gibberish**. They are confident, well-formed Korean pest names — the base model is recognizing that it's looking at insects and naming them — they just aren't among the specific 19 classes in our training set.

Top 20 off-vocab "words" from the base model (grouped by first token):

| Count | Korean | Meaning | Status |
|---:|:---|:---|:---|
| 143 | 방아벌레 | click beetle / wireworm | ✅ real pest, not in our 19 |
| 125 | 흰배나방 | white-bellied moth | ✅ real moth family |
| 115 | 방선충 | nematode (roundworm) | ✅ real plant pest |
| 94 | 벼멸구 | rice planthopper | ✅ real rice pest |
| 76 | 밤나방 | noctuid (night-moth) | ✅ real moth family |
| 61 | 나방 | "moth" (generic) | ⚠️ partial — no species |
| 49 | 배추방아벌레 | "cabbage click beetle" | ❌ invented compound |
| 32 | 애기배추방아벌레 | "baby cabbage click beetle" | ❌ invented |
| 30 | 흰날개나방 | "white-wing moth" | ❌ plausible compound |
| 25 | 방아쇠나방 | "trigger moth" | ❌ invented |
| 21 | 흰가루병 | powdery mildew | ✅ real (disease, not pest) |
| 20 | 진딧물 | aphid | ✅ real pest — not in 19 |
| 20 | 애벌레 | "larva" (generic) | ⚠️ partial |
| 19 | 흰점박이꽃무지 | white-spotted flower chafer | ✅ real beetle |
| 18 | 잎벌레 | leaf beetle (family) | ⚠️ partial |
| 15 | 방선자 | variant of 방선충 | ❌ invented |
| 11 | 벼잎벌레 | rice leaf beetle | ✅ real pest |
| 10 | 七星瓢虫 | "seven-star ladybug" | ❌ Chinese script leak |
| 6 | 대두나방 | soybean moth | ❌ plausible compound |
| 6 | 대두잎벌레 | soybean leaf beetle | ✅ real pest |

The full list of all 1,183 off-vocab raw strings is in [`evaluation/baseline_off_vocab_list.json`](https://github.com/pfox1995/pest-hyperparameter-search/blob/main/evaluation/baseline_off_vocab_list.json).

### What this reveals about what the LoRA is actually doing

The base model already has **strong Korean pest priors** — it can look at a photo of an insect and produce a confident, anatomically-plausible name. What it doesn't know is **which specific 19 Korean pest names count as valid answers for this task**.

So the LoRA's job is not to teach the model what pests look like. That knowledge is already there. The LoRA is doing **vocabulary-locking**: concentrating the model's output distribution onto a specific 19-way multiple-choice menu.

| Behavior | Base model | + LoRA adapter |
|:---|:---|:---|
| Pest recognition | Already strong (generates real pest names) | Still strong |
| Output vocabulary | Open-ended — any Korean pest name | **Locked to the 19 trained classes** |
| Off-vocabulary rate | 74.2% | **2.1%** |
| Chain-of-thought emission | Frequent (`<think>…</think>`) | Eliminated |
| Per-class F1 | Mostly 0.00 (guesses rarely land) | ≥ 0.70 for every class, ≥ 0.90 for most |

This is why **660 MB of LoRA weights on a 9B-parameter base gets us from 9% to 91% accuracy** — we are not teaching 9 billion parameters worth of pest biology. We are teaching a narrow preference for one of 19 specific Korean strings. That's a vocabulary-alignment task, which is exactly what low-rank adaptation is efficient at.

### Methodology note

The baseline numbers above are computed with **all 1,595 samples included** — off-vocab predictions counted as wrong. This is intentional; it is the honest evaluation of *"how often does the raw base model produce a correct label?"*.

For reference, if off-vocab samples are instead *dropped* from scoring (evaluating only the 412 samples where the base model happened to emit a recognizable class name), the base model reaches accuracy 36.65% / F1 weighted 27.13% — but that score is conditioned on the model having guessed within the vocabulary, which it fails to do 74% of the time.

### Reproducing

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search

# Baseline (no LoRA)
python eval_baseline.py --save-dir ./hf_eval_baseline --label baseline

# With LoRA adapter (this model)
python eval_v8.py --adapter pfox1995/pest-detector-final \
                  --save-dir ./hf_eval_v8 --label v8
```

All evaluation artifacts — confusion matrix PNGs, `metrics.json`, per-class breakdowns, full raw predictions, and the off-vocab list — are in [github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation).

---
