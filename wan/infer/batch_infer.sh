#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"

# ============================================================
# Wan 批量视频生成
# Usage:
#   bash batch_infer.sh baseline    # 完整 forward
#   bash batch_infer.sh online      # LoRA 在线推理
# ============================================================

MODE="${1:-baseline}"
PROMPT_FILE="${PROMPT_FILE:-../prompts/wan_sample_50.txt}"
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the model root}"
OUTPUT_DIR="${OUTPUT_DIR:-output/${MODE}}"
SCHEDULE_ARGS=(--replace_steps 5 6 7 9 10 11 12 13 15 16 17 18 19 21 22 23 24 25 27 28 29 30 31 33 34 35 36 37 39 40 41 42 44 46 48)
if [[ -n "${N:-}" ]]; then
    SCHEDULE_ARGS=(--N "$N")
fi

mkdir -p "${OUTPUT_DIR}"

idx=0
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
    echo "=== [${idx}] ${prompt:0:60}... ==="

    case "${MODE}" in
        baseline)
            python generate.py \
                --task t2v-1.3B --size 832*480 \
                --base_seed 42 --sample_steps "${NUM_STEPS:-50}" \
                --offload_model True --t5_cpu \
                --ckpt_dir "${CKPT_DIR}" \
                --save_file "${OUTPUT_DIR}/${idx}.mp4" \
                --prompt "${prompt}"
            ;;
        online)
            python online_gen.py \
                --task t2v-1.3B --size 832*480 \
                --base_seed 42 --frame_num 81 --sample_steps "${NUM_STEPS:-50}" \
                --offload_model True --t5_cpu \
                --ckpt_dir "${CKPT_DIR}" \
                --save_file "${OUTPUT_DIR}/${idx}.mp4" \
                --prompt "${prompt}" \
                --pred_dir "${LORA_DIR:?Set LORA_DIR to the training output}" \
                --lora_dir "${LORA_DIR:?Set LORA_DIR to the training output}" \
                --lora_rank 32 --lora_alpha 64 \
                "${SCHEDULE_ARGS[@]}"
            ;;
        *)
            echo "Unknown mode: ${MODE}. Use: baseline|online"
            exit 1
            ;;
    esac

    idx=$((idx + 1))
done < "${PROMPT_FILE}"

echo "Done. Generated ${idx} videos in ${OUTPUT_DIR}/"
