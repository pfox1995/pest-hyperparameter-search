---
base_model: unsloth/Qwen3.5-9B
library_name: transformers
model_name: best-pest-detector
tags:
- generated_from_trainer
- sft
- unsloth
- trl
licence: license
---

# Model Card for best-pest-detector

This model is a fine-tuned version of [unsloth/Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/pfox1995-none/pest-detection-hpsearch/runs/fhd287vu) 


This model was trained with SFT.

### Framework versions

- TRL: 0.22.2
- Transformers: 5.5.0
- Pytorch: 2.8.0+cu128
- Datasets: 3.6.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```