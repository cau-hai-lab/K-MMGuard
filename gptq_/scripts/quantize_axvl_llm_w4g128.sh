#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$(pwd)/outputs/hf_modules}"

MODEL_ID="${AXVL_MODEL_ID:-skt/A.X-4.0-VL-Light}"
SHARED_CACHE="${ECSO_SHARED_HF_CACHE:-/data/hai_ssh/hf_cache}"
SNAPSHOT_ROOT="$SHARED_CACHE/models--skt--A.X-4.0-VL-Light/snapshots"
if [[ "$MODEL_ID" == "skt/A.X-4.0-VL-Light" && -d "$SNAPSHOT_ROOT" ]]; then
  for snapshot in "$SNAPSHOT_ROOT"/*; do
    if [[ -d "$snapshot" ]]; then
      MODEL_ID="$snapshot"
      break
    fi
  done
fi

PYTHON_RUNNER=(python)
if [[ -n "${AXVL_GPTQ_CONDA_ENV:-}" ]]; then
  PYTHON_RUNNER=(conda run -n "$AXVL_GPTQ_CONDA_ENV" python)
fi

"${PYTHON_RUNNER[@]}" scripts/gptq_axvl_llm.py \
  --model-id "$MODEL_ID" \
  --calib-jsonl "${AXVL_CALIB_JSONL:-calib_examples/axvl_calib_64.jsonl}" \
  --output-dir "${AXVL_GPTQ_OUTPUT_DIR:-outputs/axvl_gptq_llm_w4g128}" \
  --nsamples "${AXVL_GPTQ_NSAMPLES:-64}" \
  --seqlen "${AXVL_GPTQ_SEQLEN:-2048}" \
  --wbits "${AXVL_GPTQ_WBITS:-4}" \
  --groupsize "${AXVL_GPTQ_GROUPSIZE:-128}" \
  --attn-implementation "${AXVL_GPTQ_ATTN_IMPLEMENTATION:-sdpa}" \
  --act-order \
  --sym \
  --local-files-only \
  --save-safetensors
