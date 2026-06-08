#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${ECSO_PYTHON_BIN:-python}"
STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$REPO_ROOT/outputs/hf_modules}"

IMAGE_ROOT="${ECSO_KO_IMAGE_ROOT:-/data2/AAAI/AAAI_models}"
SHARED_CACHE="${ECSO_SHARED_HF_CACHE:-/data/hai_ssh/hf_cache}"
LOCAL_CACHE="${ECSO_HF_CACHE:-outputs/hf_cache}"
OUTPUT_DIR="${ECSO_KO_OUTPUT_DIR:-outputs/mllmjailbreak_ko_ecso_gptq_v2_all}"
AXVL_MODEL="${AXVL_MODEL_PATH:-skt/A.X-4.0-VL-Light}"
TELL_MODEL="${AXVL_GPTQ_TELL_MODEL_PATH:-gptq_/outputs/axvl_gptq_llm_w4g128_v2}"
JUDGE_MODEL="${ECSO_JUDGE_MODEL_ID:-llama-3.2-3B-wildguard-ko-2410}"
LOG_DIR="${ECSO_KO_LOG_DIR:-$OUTPUT_DIR/logs}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing executable Python: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$IMAGE_ROOT/images_ko" ]]; then
    echo "Missing Korean image dataset: $IMAGE_ROOT/images_ko" >&2
    exit 1
fi
if [[ ! -d "$TELL_MODEL" ]]; then
    echo "Missing GPTQ tell model: $TELL_MODEL" >&2
    exit 1
fi
if [[ "${ECSO_LOCAL_FILES_ONLY:-1}" == "1" && ! -d "$JUDGE_MODEL" ]]; then
    echo "Missing local WildGuard judge model: $JUDGE_MODEL" >&2
    echo "Set ECSO_JUDGE_MODEL_ID to a local model path, or set ECSO_LOCAL_FILES_ONLY=0 to allow Hugging Face loading." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/${STAGE}_${RUN_ID}.log"
LATEST_LOG="$LOG_DIR/latest.log"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"

echo "Repository:       $REPO_ROOT"
echo "Stage:            $STAGE"
echo "GPU:              $CUDA_VISIBLE_DEVICES"
echo "Image root:       $IMAGE_ROOT"
echo "GPTQ tell model:  $TELL_MODEL"
echo "Judge model:      $JUDGE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log file:         $LOG_FILE"
echo
echo "Progress is printed here and written to the log file."
echo "From another terminal: tail -f $LATEST_LOG"
echo

LOCAL_FILES_ARGS=()
if [[ "${ECSO_LOCAL_FILES_ONLY:-1}" == "1" ]]; then
    LOCAL_FILES_ARGS=(--local-files-only)
fi

"$PYTHON_BIN" -u llava/eval/model_mllmjailbreak_ko_ecso_gptq.py "$STAGE" \
    --image-root "$IMAGE_ROOT" \
    --cache-dir "$LOCAL_CACHE" \
    --shared-cache-dir "$SHARED_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --model-id "$AXVL_MODEL" \
    --tell-model-path "$TELL_MODEL" \
    --judge-model-id "$JUDGE_MODEL" \
    --attn-implementation "${AXVL_ATTN_IMPLEMENTATION:-eager}" \
    --limit 0 \
    --resume \
    "${LOCAL_FILES_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$LOG_FILE"

echo
echo "Completed stage: $STAGE"
echo "Results: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
