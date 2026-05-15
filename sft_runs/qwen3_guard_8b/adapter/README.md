---
library_name: peft
license: other
base_model: model_cache/qwen3_guard_8b
tags:
- base_model:adapter:model_cache/qwen3_guard_8b
- llama-factory
- lora
- transformers
metrics:
- accuracy
pipeline_tag: text-generation
model-index:
- name: adapter
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# adapter

This model is a fine-tuned version of [model_cache/qwen3_guard_8b](https://huggingface.co/model_cache/qwen3_guard_8b) on the braveguard_sft dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1039
- Accuracy: 0.75

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 2.0337        | 3.0   | 12   | 1.1039          | 0.75     |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2