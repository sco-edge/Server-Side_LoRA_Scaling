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
