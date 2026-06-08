# Model Weights

This repository contains the code and metadata needed to reproduce the K-MMGuard GPTQ evaluation. Large model weights are handled separately because the GPTQ checkpoint is about 15 GB and individual `safetensors` shards exceed normal GitHub file limits.

## Required Models

1. Target/original VL model
   - Default: `skt/A.X-4.0-VL-Light`
   - Used for baseline generation and conditional repair generation.
   - Configure with `AXVL_MODEL_PATH`.

2. GPTQ TELL model
   - Expected default path: `gptq_/outputs/axvl_gptq_llm_w4g128_v2`
   - Used only for ECSO TELL unsafe detection.
   - Configure with `AXVL_GPTQ_TELL_MODEL_PATH`.

3. Judge model
   - Default local path used in the original run:
     `/data2/AAAI/AAAI_models/hai_zero/qwen25vl_7b/models/llama-3.2-3B-wildguard-ko-2410`
   - Used to evaluate baseline and defended responses.
   - Configure with `ECSO_JUDGE_MODEL_ID`.

## GPTQ TELL Checkpoint

The original run used:

```text
gptq_/outputs/axvl_gptq_llm_w4g128_v2
```

The checkpoint contained four large shards:

```text
model-00001-of-00004.safetensors  4.9 GB
model-00002-of-00004.safetensors  4.9 GB
model-00003-of-00004.safetensors  4.8 GB
model-00004-of-00004.safetensors  0.7 GB
```

Metadata copied from that checkpoint is stored in:

```text
weights/axvl_gptq_llm_w4g128_v2_metadata/
```

To reproduce the experiment, either:

1. Generate the checkpoint with `gptq_/scripts/quantize_axvl_llm_w4g128.sh`, or
2. Place the already generated checkpoint at `gptq_/outputs/axvl_gptq_llm_w4g128_v2`, or
3. Upload/download the checkpoint through Git LFS or a separate model host and point `AXVL_GPTQ_TELL_MODEL_PATH` to that directory.

## Why We Do Not Commit Raw Weights by Default

GitHub blocks files larger than 100 MB unless Git LFS is used. The GPTQ checkpoint is also large enough to consume LFS quota quickly. For that reason, this repo is set up to reproduce or attach weights explicitly instead of silently committing 15 GB of model files.

If the organization wants to publish the GPTQ checkpoint in this GitHub repo, run:

```bash
git lfs install
git lfs track "*.safetensors"
mkdir -p gptq_/outputs/axvl_gptq_llm_w4g128_v2
cp /path/to/axvl_gptq_llm_w4g128_v2/*.safetensors gptq_/outputs/axvl_gptq_llm_w4g128_v2/
cp weights/axvl_gptq_llm_w4g128_v2_metadata/* gptq_/outputs/axvl_gptq_llm_w4g128_v2/
git add .gitattributes gptq_/outputs/axvl_gptq_llm_w4g128_v2
git commit -m "Add GPTQ TELL checkpoint with Git LFS"
```
