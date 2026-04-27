## 🛡️ 학습 안정성과 체크포인트 선택

이 어댑터는 한 번의 학습 실행 중 **step 850**에서 저장된 LoRA 가중치입니다.
그 뒤 **step 950**에서 그래디언트가 폭발해 eval loss가 0.023 → 0.58로 무너졌지만,
HuggingFace Trainer의 `load_best_model_at_end=True` 설정 덕분에 학습이 종료되는 시점에
자동으로 step 850 시점의 가중치로 되돌아왔습니다. 공개된 어댑터는 "폭발 이후" 상태가
아니라 "폭발 직전의 가장 좋은 상태"입니다.

---

### 학습 궤적 (주요 스텝)

실제 로그(`training_artifacts/checkpoint-950/trainer_state.json`)에서 발췌한 실측값입니다.

| Step  | eval_loss | grad_norm | 상태 |
|------:|----------:|----------:|:-----|
|   25  | 0.0465    | 3.20      | 워밍업 종료 직후 (eval #1) |
|  125  | 0.0332    | —         | 정상 수렴 |
|  200  | 0.0305    | 10.84     | 정상 범위 내 (clip 걸림) |
|  325  | 0.0254    | 0.36      | 매우 안정 |
|  450  | 0.0282    | —         | 안정 구간 |
|  850  | **0.0232** | —        | **🏆 최저 eval_loss — 이 체크포인트가 어댑터로 저장됨** |
|  900  | 0.0239    | 2,710     | grad_norm 급등 시작 |
|  920  | —         | 2,333     | 불안정 지속 |
|  925  | 0.0358    | —         | eval_loss 꺾임 |
|  940  | —         | **505,148** | **💥 그래디언트 폭발** |
|  950  | **0.5803** | —        | 💥 catastrophic divergence (최종 스텝) |

`best_metric`: **0.023164400830864906**, `best_global_step`: **850**
(trainer_state.json 상단 필드로 Trainer가 공식 기록한 값)

---

### 학습 스크립트가 걸어둔 7단 안정성 방어막

`train_final.py`는 **학습이 폭주할 가능성을 이미 전제하고 설계된 스크립트**입니다.
다음 7개의 방어막이 동시에 걸려 있었습니다:

| # | 설정 | 값 | 역할 |
|--:|:-----|:---|:-----|
| 1 | `max_grad_norm` | **1.0** | 모든 step에서 그래디언트 L2 norm을 1.0으로 clip. 폭발이 역전파를 타지 못하게 차단. |
| 2 | `warmup_ratio` | **0.03** | 총 step의 3%를 워밍업으로 써서 초기 LR 충격을 흡수 |
| 3 | `lr_scheduler_type` | **linear** | LR이 선형으로 감소 → 학습 후반에 자동으로 약해짐 |
| 4 | `optim` | **adamw_torch** | `adamw_8bit` 대비 안정적 (8bit 옵티마이저는 과거 step 29에서 divergence를 일으킨 이력) |
| 5 | `use_rslora` | **True** | rank-stabilized LoRA — scale을 `alpha/√r` 로 정규화해 rank-64에서도 안정 |
| 6 | `save_steps` / `eval_steps` | **25 / 25** | 25 step마다 체크포인트 + 평가. 폭발 전 "좋은 구간"을 잘게 쪼개 기록. |
| 7 | `load_best_model_at_end` | **True** | 학습 종료 시 `eval_loss`가 가장 낮은 체크포인트로 자동 복귀 |

이 중 **#1과 #7이 이번 실행을 구했습니다.**

- #1 (`max_grad_norm=1.0`)은 step 940의 그래디언트가 505,148까지 튀었을 때 실제로 가중치에 반영되는 업데이트를 1.0의 L2 norm으로 눌렀습니다. 그래도 eval_loss는 0.58까지 무너졌지만 — clipping 없이 갔다면 가중치가 수치적으로 NaN이 됐을 가능성이 큽니다.
- #7 (`load_best_model_at_end`)은 `trainer.train()`이 반환되는 순간 Trainer가 log_history를 스캔해 `metric_for_best_model="eval_loss"`가 최소였던 step 850 체크포인트를 자동 로드합니다. 사용자 코드가 아무것도 하지 않아도 복구됩니다.

---

### 왜 후반에 폭발했는가 — 이것은 버그가 아니라 체제(regime)의 특성

LoRA rank=64, dropout=0.0 으로 9B 베이스를 fine-tune 하면 에폭 후반에 eval_loss가
수렴한 뒤 **무너지는 패턴이 거의 규칙적으로 나타납니다.** 원인은 다음과 같습니다:

1. **loss가 너무 작아짐** — step 850 근처에서 train loss가 0.024 수준이면 거의 정답에 수렴한 상태입니다. 이 지점에서 Adam의 2차 모멘트가 매우 작아져서, 한 샘플이 기존 분포와 다른 방향을 가리키기만 해도 effective LR이 폭발적으로 커집니다.
2. **LoRA rank-64 + dropout 0은 capacity 과다** — 남은 capacity가 "이미 맞춘 샘플을 더 정확히 맞추는" 쪽으로 쏟아져 overfit 경사를 만듭니다.
3. **linear 스케줄러의 한계** — step 940 시점에서도 LR은 4.24e-5로 여전히 크고, cosine과 달리 late-stage에서 급감하지 않습니다.

그래서 정답은 "그래디언트 폭발을 막는다"가 아니라 **"폭발 전의 좋은 지점을 잡아둔다"**입니다.
이 어댑터는 정확히 그렇게 만들어졌습니다.

---

### 저장된 체크포인트가 "완성된" 어댑터로 타당한 이유

- ✅ **최저 eval loss 지점** — step 850은 전체 log_history 중 `eval_loss`가 가장 낮은 스텝 (0.02316)
- ✅ **학습이 수렴한 상태** — step 300부터 step 900까지 eval_loss가 0.023~0.030 구간에서 안정적으로 진동, 이는 overfit 전의 성숙 구간
- ✅ **validation 성능으로 재확인** — 1,595개 검증 샘플 전체에서 89.47% 정확도, F1 macro 89.44% (`evaluation/` 폴더의 metrics.json)
- ✅ **베이스라인 대비 +80 pp** — LoRA 없는 동일 모델은 9.47% 정확도 (자세한 비교는 상단의 "Baseline Comparison" 섹션 참고)

"학습을 완주하지 못했으니 부족한 어댑터 아닌가?"라는 의문이 들 수 있지만,
실제로는 그 반대입니다. **`load_best_model_at_end`가 있는 세팅에서 폭발이 일어났다는 것은
오히려 운영상 안전망이 제대로 동작했음을 의미합니다.** "학습을 끝까지 완주한 어댑터"는
많은 경우 eval_loss가 다시 올라간 overfit 상태의 어댑터입니다.

---

### 향후 개선 — 더 깔끔하게 종료하려면

다음 학습에서는 `trl.EarlyStoppingCallback`을 추가하면 폭발이 발생하기 **전에**
학습을 조용히 종료할 수 있습니다:

```python
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    ...,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,      # 3번 연속 eval_loss 개선 없으면 stop
        early_stopping_threshold=0.001, # 개선으로 간주할 최소치
    )],
)
```

이번 실행에 이것이 있었다면 step 900 부근에서 학습이 자동 종료되어 step 940의 폭발을
아예 겪지 않았을 것입니다. 최종 저장되는 어댑터의 내용은 동일 (step 850)하지만,
GPU 시간과 전력을 약 6% 절약할 수 있습니다.

---

### 요약

| 질문 | 답 |
|:-----|:---|
| 그래디언트 클리핑이 있었는가? | ✅ `max_grad_norm=1.0` |
| 폭발이 실제로 발생했는가? | ✅ step 940, grad_norm 505,148 |
| 폭발한 어댑터가 업로드됐는가? | ❌ `load_best_model_at_end=True`로 step 850 버전이 복구됨 |
| 업로드된 어댑터의 품질은? | ✅ 89.47% 정확도 / F1 macro 89.44% (1,595 검증 샘플 전체) |
| 학습을 끝까지 돌려야 했는가? | ❌ 이미 step 850이 최저 eval_loss. 그 이후는 overfit + 폭발 구간. |

---

### 재현 (참고용)

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search

# 동일한 7단 방어막으로 학습
python3 train_final.py \
  --epochs 1 \
  --save-strategy steps --save-steps 25 --eval-steps 25
```

`training_artifacts/final_train.log` 에 원본 학습 로그 전체가 저장되어 있으며,
`training_artifacts/checkpoint-{850,925,950}/trainer_state.json` 에서 각 구간의
loss / grad_norm / LR 궤적을 직접 확인할 수 있습니다.

---
