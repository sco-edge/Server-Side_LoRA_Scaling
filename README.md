# Server-Side LoRA Scaling

## Overview
This project estimates the scaling factor of the aggregated LoRA adapter for each base model on the server with a single JVP, for zeroth-order federated fine-tuning across heterogeneously quantized base models.

## Base Models & Inputs
* Base Models:
   * FP16
   * GPTQ
   * HQQ
* Inputs:
   * Server-side validation data
   * Aggregated LoRA adapter (FedAvg)
* Output:
   * Scaling factor

## Requirements
* Python 3.10+
* PyTorch, Transformers, NumPy: `pip install -r requirements.txt`
* auto-gptq for GPTQ base models and hqq for HQQ base models: `pip install -r requirements-quant.txt`
* A CUDA GPU for quantized base models

## Usage
```bash
python "ZO-FL Server-Side LoRA Scaling.py" \
  --model facebook/opt-1.3b \
  --quant gptq --gptq-dir path/to/opt-1.3b-gptq-int4 \
  --adapter aggregated_adapter.pt \
  --data examples/sst2.jsonl --task sst2
```

* HQQ base model: `--quant hqq --bits 4 --group-size 32`
* fp16 base model: omit `--quant`
* `--batch-size` (default 16) and `--max-len` (default 192) set the batch size and prompt length of the forward pass.
* `--device` selects the device (default: CUDA if available).

## Adapter Format
A PyTorch `.pt` file holding a dict of tensors. For each adapted layer, `<module path>.lora_A` has shape `(r, in_features)` and `<module path>.lora_B` has shape `(out_features, r)`, where `<module path>` is the layer's name in the base model, e.g. `model.decoder.layers.0.self_attn.q_proj`. The LoRA scaling `alpha / r` is 1 by default; set `--lora-alpha` if the adapter was trained with a different `alpha`.

## Validation Data Format
A JSON Lines file with one example per line: the task's text fields and an integer `label` from `0` to `number of classes - 1`.
```json
{"sentence": "A warm, funny film that never loses its heart.", "label": 1}
```
Use `--task` for a built-in task, or `--template` and `--label-words` for any other task. See [examples](examples).
