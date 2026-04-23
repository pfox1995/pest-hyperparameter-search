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

[`unsloth/Qwen3.5-9B`](https://huggingface.co/unsloth/Qwen3.5-9B) 위에 파인튜닝된 **비전-언어 LoRA 어댑터**로, 작물 이미지로부터 **한국 농작물 해충 식별**을 수행합니다. 잎, 과실, 또는 식물 사진을 입력하면 모델은 해충의 한국어 이름을 출력하며, 해충이 없으면 `정상`을 출력합니다.

- **19개 클래스 분류기**: 해충 18종 + "정상"(해충 없음)
- **베이스 모델**: `unsloth/Qwen3.5-9B` (비전-언어, 하이브리드 linear + self attention)
- **어댑터 유형**: LoRA (PEFT), rank 64, alpha 128
- **언어**: 한국어
- **크기**: 어댑터 가중치 660 MB

---

## 성능

[Himedia-AI-01/pest-detection-korean](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean)의 **검증 세트 전체 1595개 샘플**에서 평가되었습니다.

| 지표 | 점수 |
|:---|:---:|
| **정확도(Accuracy)** | **91.36%** |
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

## 📈 기준선 비교: LoRA 어댑터 유무에 따른 차이

LoRA 파인튜닝이 모델에 실제로 무엇을 가르치는지 정량화하기 위해, **동일한 평가 파이프라인**을 어댑터가 부착되지 않은 원본 `unsloth/Qwen3.5-9B` 베이스 모델에 적용했습니다 — 같은 1,595개 검증 샘플 ([Himedia-AI-01/pest-detection-korean](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean)), 같은 전처리 (letterbox 512×512), 같은 시스템 프롬프트, 같은 디코딩 설정.

19개의 유효한 클래스 라벨 중 어느 것과도 일치하지 않는 예측은 **UNKNOWN**(어휘 외, off-vocabulary)으로 분류되며 실제 클래스에 대해 오답으로 처리됩니다. 이는 *"모델이 올바른 해충 라벨을 실제로 생성하는가?"*에 대한 솔직한 평가입니다.

### 핵심 결과

| 지표 | 베이스 Qwen3.5-9B<br>(LoRA 없음) | **LoRA 적용<br>(pest-detector-final)** | Δ |
|:---|:---:|:---:|:---:|
| **정확도(Accuracy)** | 9.47% | **89.47%** | **+80.00 pp** |
| **F1 (매크로)** | 4.32% | **89.44%** | **+85.12 pp** |
| **F1 (가중)** | 7.37% | **90.38%** | +83.01 pp |
| 정밀도 (매크로) | 8.00% | 90.88% | +82.88 pp |
| 재현율 (매크로) | 5.47% | 89.28% | +83.81 pp |
| **어휘 외 출력** | **1,183 / 1,595 (74.2%)** | **33 / 1,595 (2.1%)** | **−72.1 pp** |

### 혼동 행렬 비교

**베이스 모델 (LoRA 없음)** — 거의 모든 행이 가장 오른쪽 `UNKNOWN` 열로 흘러듭니다:

![Baseline CM](confusion_matrix_baseline_final.png)

**LoRA 어댑터 적용** — 예측이 대각선에 집중되고, `UNKNOWN` 열은 거의 비어 있습니다:

![LoRA CM](confusion_matrix_final_with_unknown.png)

---

### 베이스 모델의 출력 분석

베이스 모델이 생성한 1,183개의 어휘 외 예측을 자세히 살펴보면 놀라운 사실이 드러납니다: **이들은 의미 없는 문자열이 아닙니다**. 잘 구성된 자신감 있는 한국어 해충 이름들입니다 — 베이스 모델은 곤충을 보고 있다는 사실을 인식하고 이름을 부여하고 있습니다 — 단지 우리 학습 세트의 특정 19개 클래스에 포함되지 않을 뿐입니다.

베이스 모델의 상위 20개 어휘 외 "단어" (첫 토큰 기준 그룹화):

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

전체 1,183개 어휘 외 원시 문자열 목록은 [`evaluation/baseline_off_vocab_list.json`](https://github.com/pfox1995/pest-hyperparameter-search/blob/main/evaluation/baseline_off_vocab_list.json)에 있습니다.

---

### LoRA가 실제로 수행하는 역할

베이스 모델은 이미 **강력한 한국 해충 선행 지식**을 가지고 있습니다 — 곤충 사진을 보고 자신감 있고 해부학적으로 그럴듯한 이름을 생성할 수 있습니다. 모델이 모르는 것은 **이 작업에 대해 어떤 특정 19개 한국어 해충 이름이 유효한 답으로 간주되는지**입니다.

따라서 LoRA의 역할은 모델에게 해충이 어떻게 생겼는지 가르치는 것이 **아닙니다**. 그 지식은 이미 있습니다. LoRA가 수행하는 것은 **어휘 고정(vocabulary-locking)** 작업입니다: 모델의 출력 분포를 특정한 19지선다 메뉴에 집중시키는 것입니다.

| 동작 | 베이스 모델 | + LoRA 어댑터 |
|:---|:---|:---|
| 해충 인식 | 이미 강함 (실재 해충 이름 생성) | 여전히 강함 |
| 출력 어휘 | 개방형 — 모든 한국어 해충 이름 | **학습된 19개 클래스로 고정** |
| 어휘 외 출력 비율 | 74.2% | **2.1%** |
| 사고 연쇄(CoT) 토큰 방출 | 빈번 (`<think>…</think>`) | 제거됨 |
| 클래스별 F1 | 대부분 0.00 (추측이 거의 맞지 않음) | 모든 클래스 ≥ 0.70, 대부분 ≥ 0.90 |

이것이 바로 **9B 파라미터 베이스 모델에 660 MB의 LoRA 가중치만으로 정확도가 9%에서 91%로 향상되는 이유**입니다. 우리는 90억 파라미터 분량의 해충 생물학을 가르치고 있는 것이 아닙니다. 19개의 특정 한국어 문자열 중 하나에 대한 좁은 선호를 가르치는 것입니다. 이는 어휘 정렬 작업이며, 저순위 적응(low-rank adaptation)이 효율적으로 수행하는 바로 그 작업입니다.

---

### 방법론 주석

위의 기준선 숫자는 **1,595개 전체 샘플을 포함하여** 계산되었으며, 어휘 외 예측은 오답으로 처리되었습니다. 이는 의도적입니다 — *"원본 베이스 모델이 얼마나 자주 올바른 라벨을 생성하는가?"*에 대한 솔직한 평가입니다.

참고로, 어휘 외 샘플을 점수 계산에서 *제외*하면 (베이스 모델이 인식 가능한 클래스 이름을 생성한 412개 샘플만 평가), 베이스 모델은 정확도 36.65% / F1 가중 27.13%에 도달합니다 — 하지만 이 점수는 모델이 어휘 내에서 추측한 경우로 한정되며, 74%의 경우에 이를 실패합니다.

---

### 재현

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search

# 기준선 (LoRA 없음)
python eval_baseline.py --save-dir ./hf_eval_baseline --label baseline

# LoRA 어댑터 적용 (이 모델)
python eval_v8.py --adapter pfox1995/pest-detector-final \
                  --save-dir ./hf_eval_v8 --label v8
```

모든 평가 아티팩트 — 혼동 행렬 PNG, `metrics.json`, 클래스별 분석, 전체 원시 예측, 어휘 외 목록 — 는 [github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation)에 있습니다.

---

## 클래스 목록

모델은 19개의 한국어 라벨(해충 18종 + "정상")을 인식합니다:

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

프로덕션 환경에서는 LoRA 가중치를 베이스 모델에 한 번만 병합해 두면 추론 경로에서 PEFT가 제거되어 약 20% 빠른 순전파를 얻을 수 있습니다.

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

프로덕션 준비가 된 FastAPI 서버가 [pfox1995/pest-hyperparameter-search/serving/](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/serving)에 제공됩니다.

### 최소 배포 절차

```bash
git clone https://github.com/pfox1995/pest-hyperparameter-search.git
cd pest-hyperparameter-search/serving

pip install -r requirements.txt
python merge_adapter.py                               # 약 2분, 1회만 실행
uvicorn serve:app --host 0.0.0.0 --port 8000          # 워밍업 완료 후 준비됨
```

### 하드웨어 권장 사항

| GPU | VRAM | 비고 |
|:---|:---:|:---|
| **RTX A40** | 48 GB | **권장**. RunPod 스팟 기준 $0.22/시. bf16 + 동시 요청을 위한 충분한 여유 공간 확보. |
| RTX A6000 | 48 GB | VRAM은 동일, 클럭 약 20% 빠름, 가격은 더 비쌈. |
| RTX A5000 | 24 GB | bf16을 빠듯하게 수용, 단일 스트림 전용. |
| A100 80 GB | 80 GB | 9B 모델에는 과잉 사양 — 건너뛰세요. |

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

서버는 기본적으로 다음의 프로덕션 수준 최적화를 적용합니다:

- `stop_strings=["\n"]`로 첫 줄바꿈에서 생성 중단 (채팅 템플릿 트레일링 토큰 방지)
- 결정론적이고 캐시 가능한 응답을 위해 그리디 디코딩 + 이미지 해시 기반 LRU 캐시
- 학습 시 입력 분포와 맞추기 위해 768 px 레터박스 리사이징
- 경량 no-grad 추론을 위한 `torch.inference_mode()`

Dockerfile 및 스케일링 가이드를 포함한 전체 배포 문서는 [서빙 README](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/serving)를 참고하세요.

---

## 학습 세부 사항

- **베이스 모델**: `unsloth/Qwen3.5-9B` (Qwen3-Next 비전-언어, 하이브리드 Gated DeltaNet + self-attention)
- **어댑터**: LoRA, rank `r=64`, alpha `128`, 타겟 정규식은 `model.layers.*`와 `language_model.layers.*` 모듈 경로 모두를 포함 (`self_attn`, `linear_attn`, `mlp` — `q/k/v/o/gate/up/down/in_proj_*/out_proj`)
- **데이터셋**: [`Himedia-AI-01/pest-detection-korean`](https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean) — 한국 농작물 해충 이미지, 19개 클래스 (해충 18종 + "정상")
- **최적 체크포인트**: step 850, eval_loss = **0.0232**
- **하이퍼파라미터 탐색**: 최종 학습 전 Optuna TPE로 프록시 50 trial + 전체 10 trial 수행
- **하드웨어**: RunPod A6000 (48 GB VRAM)
- **프레임워크**: Unsloth + PEFT + TRL/SFTTrainer, bf16

학습 코드와 하이퍼파라미터 탐색 코드는 [pfox1995/pest-hyperparameter-search](https://github.com/pfox1995/pest-hyperparameter-search)에 있습니다.

---

## 한계 및 주의 사항

- **한국어 전용 출력.** 시스템 프롬프트는 모델이 한국어 해충 이름만 출력하도록 지시합니다. 다른 시스템 프롬프트를 사용하면 출력 형식이 달라질 수 있습니다.
- **고정된 19개 클래스 어휘.** 모델은 학습 세트에 있는 라벨만 출력합니다. 19개 클래스를 벗어난 드문 해충은 알려진 19개 중 하나로 잘못 분류됩니다.
- **베이스 모델 특이점 — 생성이 EOS에서 항상 멈추지 않음.** 추론 시 모델이 해충 이름 뒤에 채팅 템플릿 토큰(예: `\nassistant`)을 덧붙일 수 있습니다. `generate()`에 `stop_strings=["\n"]` 인수를 사용하면 이를 방지할 수 있습니다. 제공된 서빙 서버는 이를 자동으로 처리합니다.
- **비전 타워 민감도.** 장변 기준 768 px 이하 이미지로 학습되었습니다. 매우 고해상도 또는 매우 작은 이미지는 성능이 저하될 수 있으며, 512~768 px로 레터박스 리사이징하는 것을 권장합니다.

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
- **전체 평가 아티팩트** (CM PNG, 클래스별 지표, 원본 예측 결과, 어휘 외 목록): [github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation](https://github.com/pfox1995/pest-hyperparameter-search/tree/main/evaluation)
- **데이터셋**: https://huggingface.co/datasets/Himedia-AI-01/pest-detection-korean

---

*[Unsloth](https://unsloth.ai)로 2배 빠르게, VRAM 70% 절감하여 학습. [PEFT](https://huggingface.co/docs/peft)를 통한 LoRA 어댑터.*
