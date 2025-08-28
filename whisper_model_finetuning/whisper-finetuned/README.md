---
library_name: transformers
language:
- en
license: mit
base_model: openai/whisper-large-v3-turbo
tags:
- generated_from_trainer
datasets:
- krishan23/indian_english
metrics:
- wer
model-index:
- name: Whisper Indian Acccent
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: indian english accent
      type: krishan23/indian_english
      args: 'split: train'
    metrics:
    - name: Wer
      type: wer
      value: 4.390847247990106
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper Indian Acccent

This model is a fine-tuned version of [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) on the indian english accent dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1700
- Wer: 4.3908

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
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss | Wer    |
|:-------------:|:-------:|:----:|:---------------:|:------:|
| 0.0425        | 2.6247  | 1000 | 0.1022          | 5.5246 |
| 0.0176        | 5.2493  | 2000 | 0.1252          | 5.5040 |
| 0.0038        | 7.8740  | 3000 | 0.1524          | 5.1433 |
| 0.0008        | 10.4987 | 4000 | 0.1628          | 4.3393 |
| 0.0003        | 13.1234 | 5000 | 0.1700          | 4.3908 |


### Framework versions

- Transformers 4.48.3
- Pytorch 2.2.0a0+81ea7a4
- Datasets 3.3.0
- Tokenizers 0.21.0
