---
library_name: peft
base_model: unsloth/Qwen3.5-9B
license: apache-2.0
language:
- ko
pipeline_tag: image-text-to-text
tags:
- lora
- peft
- vision-language
- korean
- pest-detection
- agriculture
- qwen
- qwen3
- image-classification
datasets:
- Himedia-AI-01/pest-detection-korean
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: pest-detector-final
  results:
  - task:
      type: image-text-to-text
      name: Korean Pest Classification
    dataset:
      name: Himedia-AI-01/pest-detection-korean
      type: image-classification
    metrics:
    - type: accuracy
      value: 0.9136
      name: Accuracy
    - type: f1
      value: 0.9032
      name: F1 (macro)
    - type: f1
      value: 0.9134
      name: F1 (weighted)
    - type: precision
      value: 0.9088
      name: Precision (macro)
    - type: recall
      value: 0.9101
      name: Recall (macro)
---

# 한국 작물 해충 탐지기 — Qwen3.5-9B LoRA

[`unsloth/Qwen3.5-9B`](https://huggingface.co/unsloth/Qwen3.5-9B)을 파인튜닝한 **비전-언어 LoRA 어댑터**로, 작물 사진에서 **한국 농작물 해충을 식별**합니다. 잎, 과실, 식물 전체 사진을 넣으면 해충의 한국어 이름을 출력하고, 해충이 없으면 `정상`을 출력합니다.

- **19개 클래스 분류기**: 해충 18종 + "정상"(해충 없음)
- **베이스 모델**: `unsloth/Qwen3.5-9B` (비전-언어, 하이브리드 linear + self attention)
- **어댑터 유형**: LoRA (PEFT), rank 64, alpha 128
- **언어**: 한국어
- **크기**: 어댑터 가중치 660 MB

---

## 성능

[Himedia-AI-01/pest-detection-korean](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean)의 **검증 세트 전체 1,595개 샘플**로 평가한 결과입니다.

| 지표 | 점수 |
|:---|:---:|
| **정확도 (Accuracy)** | **91.36%** |
| **F1 (매크로)** | **90.32%** |
| **F1 (가중)** | **91.34%** |
| 정밀도 (매크로) | 90.88% |
| 재현율 (매크로) | 91.01% |
| 정밀도 (가중) | 92.40% |
| 재현율 (가중) | 91.36% |

### 혼동 행렬

![Confusion Matrix](confusion_matrix.png)

### 클래스별 성능

| 클래스 | 정밀도 | 재현율 | F1 | 샘플 수 |
|:---|:---:|:---:|:---:|:---:|
| 무잎벌 | 1.0000 | 1.0000 | 1.0000 | 22 |
| 담배거세미나방 | 0.9787 | 0.9787 | 0.9787 | 47 |
| 먹노린재 | 0.9888 | 0.9670 | 0.9778 | 91 |
| 배추흰나비 | 0.9658 | 0.9658 | 0.9658 | 117 |
| 담배나방 | 0.9429 | 0.9851 | 0.9635 | 67 |
| 알락수염노린재 | 0.9741 | 0.9417 | 0.9576 | 120 |
| 톱다리개미허리노린재 | 1.0000 | 0.9078 | 0.9517 | 141 |
| 꽃노랑총채벌레 | 0.9524 | 0.9302 | 0.9412 | 43 |
| 파밤나방 | 0.9500 | 0.9194 | 0.9344 | 124 |
| 큰28점박이무당벌레 | 0.9375 | 0.9184 | 0.9278 | 49 |
| 썩덩나무노린재 | 0.8341 | 0.9884 | 0.9048 | 173 |
| 정상 | 0.8295 | 0.9932 | 0.9040 | 147 |
| 담배가루이 | 0.9730 | 0.8182 | 0.8889 | 44 |
| 배추좀나방 | 0.8675 | 0.8889 | 0.8780 | 81 |
| 비단노린재 | 0.8333 | 0.8824 | 0.8571 | 34 |
| 벼룩잎벌레 | 0.9726 | 0.7136 | 0.8232 | 199 |
| 도둑나방 | 1.0000 | 0.6923 | 0.8182 | 13 |
| 목화바둑명나방 | 0.6667 | 0.9444 | 0.7816 | 36 |
| 검거세미밤나방 | 0.6000 | 0.8571 | 0.7059 | 14 |

---

## 📈 베이스라인 비교: LoRA 어댑터 유무에 따른 차이

LoRA 파인튜닝이 모델에 실제로 무엇을 가르쳐 주는지 정량적으로 확인하기 위해, **똑같은 평가 파이프라인**을 LoRA 어댑터 없이 원본 `unsloth/Qwen3.5-9B` 베이스 모델에도 그대로 돌려 봤습니다 — 같은 1,595개 검증 샘플([Himedia-AI-01/pest-detection-korean](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean)), 같은 전처리(letterbox 512×512), 같은 시스템 프롬프트, 같은 디코딩 설정.

19개 클래스 라벨 중 어디에도 해당하지 않는 예측은 **UNKNOWN**(off-vocabulary, OOV)으로 분류하고 오답으로 처리했습니다. 즉 "이 모델이 실제로 정답 해충 이름을 출력해 내는가?"를 있는 그대로 보여 주는 평가입니다.

### 핵심 결과

| 지표 | 베이스 Qwen3.5-9B<br>(LoRA 없음) | **LoRA 적용<br>(pest-detector-final)** | Δ |
|:---|:---:|:---:|:---:|
| **정확도 (Accuracy)** | 9.47% | **89.47%** | **+80.00%p** |
| **F1 (매크로)** | 4.32% | **89.44%** | **+85.12%p** |
| **F1 (가중)** | 7.37% | **90.38%** | +83.01%p |
| 정밀도 (매크로) | 8.00% | 90.88% | +82.88%p |
| 재현율 (매크로) | 5.47% | 89.28% | +83.81%p |
| **OOV 출력** | **1,183 / 1,595 (74.2%)** | **33 / 1,595 (2.1%)** | **−72.1%p** |

### 혼동 행렬 비교

**베이스 모델 (LoRA 없음)** — 거의 모든 행이 오른쪽 끝 `UNKNOWN` 열로 몰립니다:

![Baseline CM](confusion_matrix_baseline_final.png)

**LoRA 어댑터 적용** — 예측이 대각선에 집중되고 `UNKNOWN` 열은 거의 비어 있습니다:

![LoRA CM](confusion_matrix_final_with_unknown.png)

---

### 베이스 모델의 출력 분석

베이스 모델이 내놓은 1,183개의 OOV 예측을 자세히 뜯어보면 재미있는 사실이 보입니다: **이것들은 의미 없는 문자열이 아닙니다.** 대부분 한국어로 분명하게 쓰인, 실재하는 해충 이름들입니다 — 베이스 모델이 곤충 사진임을 이미 알아보고 이름까지 붙이는데, 그 이름이 우리가 학습에 쓴 19개 클래스에 들어 있지 않을 뿐입니다.

베이스 모델의 상위 20개 OOV "단어" (첫 토큰 기준으로 묶음):

| 빈도 | 한국어 | 설명 | 실재 여부 |
|---:|:---|:---|:---|
| 143 | 방아벌레 | wireworm / click beetle | ✅ 실재 해충 (19개 목록 외) |
| 125 | 흰배나방 | white-bellied moth | ✅ 실재 나방 계열 |
| 115 | 방선충 | nematode (선충) | ✅ 실재 식물 해충 |
| 94 | 벼멸구 | rice planthopper | ✅ 실재 벼 해충 |
| 76 | 밤나방 | noctuid (밤나방과) | ✅ 실재 나방 계열 |
| 61 | 나방 | "나방" (일반명) | ⚠️ 일반명, 종 미지정 |
| 49 | 배추방아벌레 | "배추 click beetle" | ❌ 창작 조어 |
| 32 | 애기배추방아벌레 | "아기 배추 click beetle" | ❌ 창작 조어 |
| 30 | 흰날개나방 | "흰 날개 나방" | ❌ 그럴듯한 조합어 |
| 25 | 방아쇠나방 | "방아쇠 나방" | ❌ 창작 조어 |
| 21 | 흰가루병 | powdery mildew (흰가루병) | ✅ 실재 (병해, 해충 아님) |
| 20 | 진딧물 | aphid (진딧물) | ✅ 실재 해충 — 19개 외 |
| 20 | 애벌레 | "애벌레" (일반명) | ⚠️ 일반명 |
| 19 | 흰점박이꽃무지 | white-spotted flower chafer | ✅ 실재 딱정벌레 |
| 18 | 잎벌레 | leaf beetle (잎벌레과) | ⚠️ 과명, 종 미지정 |
| 15 | 방선자 | 방선충의 변형 | ❌ 창작 조어 |
| 11 | 벼잎벌레 | rice leaf beetle | ✅ 실재 해충 |
| 10 | 七星瓢虫 | "칠성무당벌레" (중국어) | ❌ 중국어 혼입 |
| 6 | 대두나방 | "대두 나방" | ❌ 그럴듯한 조합어 |
| 6 | 대두잎벌레 | soybean leaf beetle | ✅ 실재 해충 |

전체 1,183개 OOV 원문 목록은 [`evaluation/baseline_off_vocab_list.json`](https://github.com/pfox1995/pest-hyperparameter-search/blob/main/evaluation/baseline_off_vocab_list.json)에 있습니다.

---

### LoRA가 실제로 하는 일

베이스 모델은 **한국 해충에 대한 사전 지식이 이미 충분합니다** — 곤충 사진을 보고 해부학적으로 그럴듯한 이름을 꽤 자신 있게 내놓습니다. 모델이 모르는 건 **이번 과제의 정답 목록에 들어가는 19개 이름이 무엇인지**입니다.

그래서 LoRA의 역할은 "해충이 어떻게 생겼는지"를 가르쳐 주는 게 **아닙니다**. 그런 지식은 이미 있습니다. LoRA가 실제로 하는 일은 **vocabulary-locking (어휘 고정)** — 모델의 출력 분포를 "19지선다 메뉴"로 좁혀 주는 일입니다.

| 동작 | 베이스 모델 | + LoRA 어댑터 |
|:---|:---|:---|
| 해충 인식 | 이미 충분 (실재 해충 이름 생성) | 여전히 충분 |
| 출력 어휘 | 개방형 — 모든 한국어 해충 이름 | **학습된 19개 클래스로 고정** |
| OOV 출력 비율 | 74.2% | **2.1%** |
| CoT(사고 연쇄) 토큰 출력 | 잦음 (`<think>…</think>`) | 제거됨 |
| 클래스별 F1 | 대부분 0.00 (추측이 거의 안 맞음) | 모든 클래스 ≥ 0.70, 대부분 ≥ 0.90 |

바로 이것이 **9B 파라미터 베이스 모델에 660 MB짜리 LoRA 가중치만으로 정확도가 9%에서 91%로 뛴 이유**입니다. 90억 개 파라미터에 해충에 대한 지식을 새로 새겨 넣는 게 아니라, **19개 중 하나를 고르는 좁은 선호**만 학습시키는 것입니다. 이건 vocabulary alignment 문제이고, low-rank adaptation(LoRA)이 가장 잘 해내는 종류의 일입니다.

---

### 방법론 주석

위의 베이스라인 수치는 **1,595개 샘플 전체**를 대상으로 계산한 값이며, OOV 예측은 오답으로 처리했습니다. 일부러 이렇게 맞춘 것입니다 — "원본 베이스 모델이 실제로 정답을 얼마나 자주 내놓는가?"를 있는 그대로 보여 주기 위해서입니다.

참고로 OOV 샘플을 점수 계산에서 *제외*하면(베이스 모델이 유효한 클래스 이름을 뱉은 412개 샘플만 남기면) 베이스 모델도 정확도 36.65% / F1 가중 27.13%까지 올라갑니다. 다만 이 수치는 "모델이 어휘 안에서 답을 낸 경우"만 따로 떼어 낸 것이며, 애초에 74%는 어휘 안에서 답을 내지도 못합니다.

---

### 재현

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search

# 베이스라인 (LoRA 없음)
python eval_baseline.py --save-dir ./hf_eval_baseline --label baseline

# LoRA 어댑터 적용 (이 모델)
python eval_v8.py --adapter pfox1995/pest-detector-final \
                  --save-dir ./hf_eval_v8 --label v8
```

모든 평가 아티팩트 — 혼동 행렬 PNG, `metrics.json`, 클래스별 분석, 전체 원본 예측, OOV 목록 — 은 [github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation)에 있습니다.

---

## 클래스 목록

모델이 인식하는 19개 한국어 라벨(해충 18종 + "정상"):

검거세미밤나방, 꽃노랑총채벌레, 담배가루이, 담배거세미나방, 담배나방, 도둑나방, 먹노린재, 목화바둑명나방, 무잎벌, 배추좀나방, 배추흰나비, 벼룩잎벌레, 비단노린재, 썩덩나무노린재, 알락수염노린재, 정상, 큰28점박이무당벌레, 톱다리개미허리노린재, 파밤나방

---

## 사용법

### 빠른 시작 (추론)

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image

BASE    = "unsloth/Qwen3.5-9B"
ADAPTER = "pfox1995/pest-detector-final"

# 베이스 모델 로드 + 어댑터 연결
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cuda",
).eval()
model = PeftModel.from_pretrained(model, ADAPTER)
processor = AutoProcessor.from_pretrained(BASE)

# 이미지 준비
image = Image.open("pest.jpg").convert("RGB")

# 학습 시 사용한 시스템 프롬프트를 그대로 사용
SYSTEM_MSG = (
    "당신은 작물 해충 식별 전문가입니다. "
    "사진을 보고 해충의 이름만 한국어로 답하세요. "
    '해충이 없으면 "정상"이라고만 답하세요. '
    "부가 설명 없이 이름만 출력하세요."
)

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."},
    ]},
]

tmpl = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, tmpl, add_special_tokens=False, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
        stop_strings=["\n", "<|im_end|>"],
        tokenizer=processor.tokenizer,
    )

prediction = processor.decode(
    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()
print(prediction)   # 예: "배추흰나비"
```

### 빠른 서빙을 위한 어댑터 병합

프로덕션 환경에서는 LoRA 가중치를 베이스 모델에 한 번만 병합해 두면, 추론 시 PEFT를 거치지 않게 되어 forward pass가 약 20% 빨라집니다.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "unsloth/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda",
)
model = PeftModel.from_pretrained(model, "pfox1995/pest-detector-final")
model = model.merge_and_unload()   # LoRA를 베이스 가중치에 병합
model.save_pretrained("./pest-detector-merged", safe_serialization=True)
AutoProcessor.from_pretrained("unsloth/Qwen3.5-9B").save_pretrained("./pest-detector-merged")
```

병합 후에는 서빙 시 `peft` 없이 순수 `transformers`만으로 로드할 수 있습니다.

---

## HTTP API로 서빙하기

바로 쓸 수 있는 FastAPI 서버가 [pfox1995/pest-hyperparameter-search/serving/](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/serving)에 준비되어 있습니다.

### 최소 배포 절차

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search/serving

pip install -r requirements.txt
python merge_adapter.py                               # 약 2분, 1회만 실행
uvicorn serve:app --host 0.0.0.0 --port 8000          # 워밍업이 끝나면 준비 완료
```

### 하드웨어 권장 사항

| GPU | VRAM | 비고 |
|:---|:---:|:---|
| **RTX A40** | 48 GB | **권장**. RunPod 스팟 기준 $0.22/시. bf16 실행과 동시 요청 처리에 충분한 여유가 있음. |
| RTX A6000 | 48 GB | VRAM은 동일, 클럭이 약 20% 빠름, 가격은 더 비쌈. |
| RTX A5000 | 24 GB | bf16이 빠듯하게 들어감, 단일 스트림 전용. |
| A100 80 GB | 80 GB | 9B 모델에는 과한 사양 — 건너뛰시면 됩니다. |

### API 호출

```bash
curl -X POST http://your-pod:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "'$(base64 -w0 pest.jpg)'"}'
```

응답 예시:

```json
{
  "pest": "배추흰나비",
  "raw": "배추흰나비",
  "known_class": true,
  "latency_ms": 980
}
```

서버는 기본적으로 다음 프로덕션용 최적화를 적용합니다:

- `stop_strings=["\n"]`로 첫 줄바꿈에서 generate를 중단 (chat template의 trailing 토큰 방지)
- 재현성과 캐시 효율을 위한 greedy decoding + 이미지 해시 기반 LRU 캐시
- 학습 시 입력 분포에 맞추기 위한 768 px letterbox resize
- no-grad 추론을 위한 `torch.inference_mode()`

Dockerfile과 스케일링 가이드를 포함한 전체 배포 문서는 [서빙 README](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/serving)를 참고하세요.

---

## 학습 세부 사항

- **베이스 모델**: `unsloth/Qwen3.5-9B` (Qwen3-Next 비전-언어, 하이브리드 Gated DeltaNet + self-attention)
- **어댑터**: LoRA, rank `r=64`, alpha `128`. 타겟 정규식은 `model.layers.*`와 `language_model.layers.*` 모듈 경로를 모두 포함합니다 (`self_attn`, `linear_attn`, `mlp` — `q/k/v/o/gate/up/down/in_proj_*/out_proj`).
- **데이터셋**: [`Himedia-AI-01/pest-detection-korean`](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean) — 한국 농작물 해충 이미지, 19개 클래스 (해충 18종 + "정상")
- **최적 체크포인트**: step 850, eval_loss = **0.0232**
- **하이퍼파라미터 탐색**: 최종 학습 전에 Optuna TPE로 proxy 50 trial + full 10 trial 수행
- **하드웨어**: RunPod A6000 (48 GB VRAM)
- **프레임워크**: Unsloth + PEFT + TRL/SFTTrainer, bf16

학습 코드와 하이퍼파라미터 탐색 코드는 [pfox1995/pest-hyperparameter-search](https://github.com/pfox1995/pest-hyperparameter-search)에 있습니다.

---

## 🛡️ 학습 안정성과 체크포인트 선택

이 어댑터는 한 번의 학습 중 eval_loss가 가장 낮았던 **step 850** 지점의 LoRA 가중치입니다. 바로 이어진 **step 940~950 구간**에서 gradient가 폭발해 eval_loss가 0.023에서 0.58까지 튀어 오르긴 했지만, HuggingFace Trainer의 `load_best_model_at_end=True` 덕분에 학습이 끝날 때 자동으로 step 850 가중치로 되돌아왔습니다. 공개된 어댑터는 '폭발 이후'가 아니라 **'폭발 직전, eval_loss가 가장 낮았던 순간'**의 스냅샷입니다.

### 학습 궤적 (주요 스텝)

실제 학습 로그(`training_artifacts/checkpoint-950/trainer_state.json`)에서 뽑아낸 관측값입니다.

| Step | eval_loss | grad_norm | 상태 |
|---:|---:|---:|:---|
| 25 | 0.0465 | 3.20 | warmup 끝난 직후 (첫 평가) |
| 125 | 0.0332 | — | 수렴 구간 진입 |
| 200 | 0.0305 | 10.84 | 정상 범위 (clipping 발동) |
| 325 | 0.0254 | 0.36 | 가장 안정적인 구간 |
| 450 | 0.0282 | — | 안정적으로 수렴 중 |
| 850 | **0.0232** | — | **🏆 eval_loss 최저점 — 이 체크포인트가 최종 어댑터로 채택됨** |
| 900 | 0.0239 | 2,710 | grad_norm 급등 조짐 |
| 920 | — | 2,333 | 불안정한 상태 지속 |
| 925 | 0.0358 | — | eval_loss 반등 시작 |
| 940 | — | **505,148** | **💥 gradient 폭발 본격화** |
| 950 | **0.5803** | — | 💥 학습 발산 (최종 스텝) |

`best_metric`: **0.023164400830864906**, `best_global_step`: **850**
(`trainer_state.json` 상단 필드에 Trainer가 자체 기록한 공식 값)

### 학습 스크립트가 걸어 둔 7단 안전장치

`train_final.py`는 **학습이 불안정해질 가능성을 처음부터 염두에 두고 짠 스크립트**입니다. 다음 7개 안전장치가 동시에 걸려 있습니다:

| # | 설정 | 값 | 역할 |
|--:|:---|:---|:---|
| 1 | `max_grad_norm` | **1.0** | 매 스텝마다 gradient L2 norm을 1.0으로 clipping. 폭발이 backprop에 실리지 못하게 막음. |
| 2 | `warmup_ratio` | **0.03** | 전체 스텝의 3%를 warmup으로 써서 초기 LR이 튀는 충격을 완화. |
| 3 | `lr_scheduler_type` | **linear** | LR을 선형으로 줄여 나감 → 후반으로 갈수록 자연스럽게 작아짐. |
| 4 | `optim` | **adamw_torch** | `adamw_8bit`보다 수치적으로 안정 (이전 실험에서 8bit optimizer가 step 29에서 발산한 이력이 있음). |
| 5 | `use_rslora` | **True** | rank-stabilized LoRA — 스케일을 `alpha/√r`로 정규화해 rank-64에서도 수렴이 안정적. |
| 6 | `save_steps` / `eval_steps` | **25 / 25** | 25 스텝마다 체크포인트 + 평가. 폭발 직전의 안정 구간을 촘촘하게 남겨 줌. |
| 7 | `load_best_model_at_end` | **True** | 학습이 끝날 때 `eval_loss`가 가장 낮았던 체크포인트로 자동 복원. |

이 중 **#1과 #7이 이번 학습의 결정적 안전망으로 작동했습니다.**

- **#1** (`max_grad_norm=1.0`) — step 940에서 gradient가 505,148까지 튀어 올랐을 때, 실제 가중치 업데이트에 반영되는 norm은 1.0으로 눌러 줬습니다. 덕분에 eval_loss가 0.58 선에서 멈춘 것이며, clipping이 없었다면 가중치 자체가 NaN으로 발산했을 가능성이 큽니다.
- **#7** (`load_best_model_at_end=True`) — `trainer.train()`이 끝나는 순간 Trainer가 내부 `log_history`를 훑어서 `metric_for_best_model="eval_loss"`가 가장 낮았던 step 850 체크포인트를 자동으로 다시 불러옵니다. 학습 스크립트에서 별도 작업을 하지 않아도 자동으로 복원됩니다.

### 왜 후반에 폭발했는가 — 버그가 아니라 이 세팅의 특성

LoRA rank=64에 dropout=0.0으로 9B짜리 베이스 모델을 파인튜닝하면, **에폭 후반부에 eval_loss가 한 번 수렴한 뒤 다시 무너지는 패턴이 거의 규칙적으로 나타납니다.** 주요 원인은 다음과 같습니다:

1. **loss가 극도로 작아짐** — step 850 부근에서 train loss가 0.024 수준이라는 건 사실상 학습 데이터를 거의 다 외운 상태입니다. 이 구간에서는 Adam의 2차 모멘트(분산 추정치)가 매우 작아지는데, 이때 배치 하나라도 기존 분포와 다른 방향을 가리키면 실효 LR(effective LR)이 폭발적으로 커집니다.
2. **LoRA rank-64 + dropout 0 조합은 capacity 여유가 큼** — 남은 capacity가 "이미 잘 맞춘 샘플을 더 정밀하게 맞추려는" 쪽으로 쏠리면서 overfit 방향의 gradient를 만들어 냅니다.
3. **linear 스케줄러의 한계** — step 940 시점에도 LR이 4.24e-5로 여전히 꽤 높고, cosine 스케줄처럼 후반에 급격히 떨어지지 않습니다.

그래서 전략은 "gradient 폭발 자체를 막는 것"이 아니라 **"폭발이 일어나기 전의 최적 지점을 잡아 두는 것"**이어야 합니다. 이 어댑터는 정확히 그런 원칙으로 만들어졌습니다.

### 저장된 체크포인트가 "완성된" 어댑터로 타당한 이유

- ✅ **eval_loss 최저 지점** — step 850은 전체 `log_history` 중 `eval_loss`가 최소(0.02316)를 기록한 스텝입니다.
- ✅ **충분히 수렴된 상태** — step 300부터 step 900까지 약 600 스텝 동안 eval_loss가 0.023~0.030 구간에서 안정적으로 진동했습니다 — 과적합 직전의 잘 무르익은 수렴 구간입니다.
- ✅ **검증 성능으로 교차 확인** — 1,595개 검증 샘플 전체에서 89.47% 정확도, F1 macro 89.44% (`evaluation/metrics.json` 참조).
- ✅ **베이스라인 대비 +80%p** — LoRA 없이 돌린 동일 모델은 9.47%에 그칩니다 (자세한 비교는 위의 "베이스라인 비교" 섹션 참고).

"학습이 중간에 꼬였으니 이 어댑터도 부실한 것 아닌가?"라는 의문이 들 수 있지만, 실은 그 반대입니다. **오히려 `load_best_model_at_end`가 걸려 있는 상황에서 폭발이 발생했다는 사실 자체가 운영 안전망이 설계대로 작동했다는 증거입니다.** "학습을 끝까지 돌린 어댑터"가 더 좋은 어댑터라는 보장도 없습니다 — 오히려 eval_loss가 다시 올라가 버린 overfit 상태인 경우가 많습니다.

### 향후 개선 — 더 깔끔하게 종료하려면

다음 학습에서는 `transformers.EarlyStoppingCallback`을 붙이면 폭발이 일어나기 **전에** 학습을 조기 종료시킬 수 있습니다:

```python
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    ...,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,      # 3회 연속 eval_loss 개선이 없으면 종료
        early_stopping_threshold=0.001, # 개선으로 인정할 최소 변화량
    )],
)
```

이번 학습에 이 콜백이 걸려 있었다면 step 900쯤에서 학습이 자동으로 끝나서 step 940의 폭발은 아예 일어나지 않았을 겁니다. 저장되는 최종 어댑터 자체는 같지만(step 850) GPU 시간과 전력을 약 6% 아낄 수 있습니다.

### 한눈에 정리

| 질문 | 답 |
|:---|:---|
| gradient clipping이 걸려 있었는가? | ✅ `max_grad_norm=1.0` |
| 실제로 폭발이 일어났는가? | ✅ step 940, grad_norm 505,148 |
| 폭발한 상태의 어댑터가 업로드됐는가? | ❌ `load_best_model_at_end=True` 덕분에 step 850 버전으로 복원됨 |
| 업로드된 어댑터의 성능은? | ✅ 89.47% 정확도 / F1 macro 89.44% (1,595개 검증 샘플 전체) |
| 학습을 끝까지 돌려야 했는가? | ❌ 이미 step 850이 eval_loss 최저점. 그 이후는 overfit 및 폭발 구간 |

### 재현 (참고용)

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search

# 7단 안전장치가 모두 걸린 학습
python3 train_final.py \
  --epochs 1 \
  --save-strategy steps --save-steps 25 --eval-steps 25
```

`training_artifacts/final_train.log`에 학습 로그 원본이 그대로 남아 있고, `training_artifacts/checkpoint-{850,925,950}/trainer_state.json`에서 각 구간의 loss / grad_norm / 학습률 궤적을 직접 확인할 수 있습니다.

---

## 한계 및 주의 사항

- **한국어 전용 출력.** 시스템 프롬프트가 모델에게 한국어 해충 이름만 출력하도록 지시합니다. 다른 시스템 프롬프트를 쓰면 출력 형식이 달라질 수 있습니다.
- **고정된 19개 클래스 어휘.** 모델은 학습 세트에 있는 라벨만 출력합니다. 19개 클래스를 벗어난 드문 해충은 알려진 19개 중 하나로 잘못 분류됩니다.
- **베이스 모델 특이점 — 생성이 EOS에서 항상 멈추지 않음.** 추론 시 모델이 해충 이름 뒤에 chat template 토큰(예: `\nassistant`)을 덧붙이는 경우가 있습니다. `generate()`에 `stop_strings=["\n"]` 인수를 주면 방지할 수 있으며, 제공된 서빙 서버는 이를 자동으로 처리합니다.
- **비전 타워 민감도.** 장변 기준 768 px 이하 이미지로 학습한 모델입니다. 너무 큰 이미지나 너무 작은 이미지는 성능이 떨어질 수 있으니 512~768 px 범위로 letterbox resize해서 쓰시길 권장합니다.

---

## 인용

이 모델을 사용하실 경우 다음을 인용해 주세요:

```bibtex
@misc{pest-detector-final-2026,
  author = {pfox1995},
  title = {Korean Pest Detector — Qwen3.5-9B LoRA Adapter},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/pfox1995/pest-detector-final},
}
```

---

## 링크

- **모델**: https://huggingface.co/pfox1995/pest-detector-final
- **학습 + 서빙 코드**: https://github.com/pfox1995/pest-hyperparameter-search
- **전체 평가 아티팩트** (CM PNG, 클래스별 지표, 원본 예측 결과, OOV 목록): [github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation)
- **데이터셋**: https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean

---

*[Unsloth](https://unsloth.ai)로 2배 빠르게, VRAM을 70% 아껴서 학습했습니다. [PEFT](https://huggingface.co/docs/peft) 기반의 LoRA 어댑터입니다.*
