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
