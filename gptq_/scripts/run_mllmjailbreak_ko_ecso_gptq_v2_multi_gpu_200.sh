#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_LIST="${ECSO_KO_GPUS:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_SHARDS="${#GPUS[@]}"
TOTAL_LIMIT="${ECSO_KO_TOTAL_LIMIT:-200}"
OUTPUT_DIR="${ECSO_KO_OUTPUT_DIR:-outputs/mllmjailbreak_ko_ecso_gptq_v2_multi_gpu_200}"
SHARD_ROOT="$OUTPUT_DIR/shards"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
    echo "ECSO_KO_GPUS must contain at least one GPU index" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$SHARD_ROOT"

pids=()
for shard_index in "${!GPUS[@]}"; do
    gpu="${GPUS[$shard_index]}"
    shard_dir="$SHARD_ROOT/shard_${shard_index}"
    mkdir -p "$shard_dir"
    echo "Starting shard $((shard_index + 1))/$NUM_SHARDS on GPU $gpu: $shard_dir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    ECSO_KO_OUTPUT_DIR="$shard_dir" \
    bash gptq_/scripts/run_mllmjailbreak_ko_ecso_gptq_v2_all.sh all \
        --total-limit "$TOTAL_LIMIT" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard_index" \
        > "$shard_dir/launcher.log" 2>&1 &
    pids+=("$!")
done

status=0
for shard_index in "${!pids[@]}"; do
    if wait "${pids[$shard_index]}"; then
        echo "Completed shard $((shard_index + 1))/$NUM_SHARDS on GPU ${GPUS[$shard_index]}"
    else
        echo "Failed shard $((shard_index + 1))/$NUM_SHARDS on GPU ${GPUS[$shard_index]}" >&2
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one shard failed. Check $SHARD_ROOT/shard_*/launcher.log" >&2
    exit "$status"
fi

for filename in selected.jsonl baseline.jsonl tell.jsonl defended.jsonl judged.jsonl; do
    : > "$OUTPUT_DIR/$filename"
    for shard_index in "${!GPUS[@]}"; do
        cat "$SHARD_ROOT/shard_${shard_index}/$filename" >> "$OUTPUT_DIR/$filename"
    done
done

echo "Merged $TOTAL_LIMIT samples. Creating combined summary."
ECSO_KO_OUTPUT_DIR="$OUTPUT_DIR" \
bash gptq_/scripts/run_mllmjailbreak_ko_ecso_gptq_v2_all.sh summarize

echo "Completed multi-GPU evaluation: $OUTPUT_DIR"
