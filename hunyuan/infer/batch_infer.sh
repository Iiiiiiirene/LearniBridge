#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"

# ============================================================
# HunyuanVideo 批量推理 (从 prompt 文件逐条生成)
# Usage:
#   bash batch_infer.sh online      # LoRA 在线推理
#   bash batch_infer.sh baseline    # 完整 forward
# ============================================================

MODE="${1:-online}"
PROMPT_FILE="${PROMPT_FILE:-../prompts/sample_50.txt}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODE}}"
SCHEDULE_ARGS=(--replace-steps 2 3 4 5 6 7 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26 27 28 30 31 32 33 34 35 37 38 39 40 41 42 44 45 46 47 48 49)
if [[ -n "${N:-}" ]]; then
    SCHEDULE_ARGS=(--N "$N")
fi

while IFS= read -r prompt; do
    [[ -z "$prompt" ]] && continue
    echo ">>> ${prompt:0:60}..."

    case "${MODE}" in
        online)
            CUDA_VISIBLE_DEVICES=$GPU python3 online_lora.py \
                --model-base "${MODEL_BASE:?Set MODEL_BASE to the model root}" \
                --dit-weight "${DIT_WEIGHT:-$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt}" \
                --lora-dir "${LORA_DIR:?Set LORA_DIR to the training output}" \
                "${SCHEDULE_ARGS[@]}" \
                --block-idx 39 \
                --target-modules auto \
                --lora-rank 32 --lora-alpha 64.0 \
                --video-size 544 960 --video-length 49 \
                --infer-steps "${NUM_STEPS:-50}" \
                --prompt "$prompt" \
                --seed 42 --flow-reverse --use-cpu-offload \
                --save-path "$OUTPUT_DIR"
            ;;
        baseline)
            CUDA_VISIBLE_DEVICES=$GPU python3 sample_video.py \
                --model-base "${MODEL_BASE:?Set MODEL_BASE to the model root}" \
                --dit-weight "${DIT_WEIGHT:-$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt}" \
                --video-size 544 960 --video-length 49 \
                --infer-steps "${NUM_STEPS:-50}" \
                --prompt "$prompt" \
                --seed 42 --embedded-cfg-scale 6.0 --flow-shift 7.0 \
                --flow-reverse --use-cpu-offload \
                --save-path "$OUTPUT_DIR"
            ;;
        *)
            echo "Unknown mode: ${MODE}. Use: online|baseline"
            exit 1
            ;;
    esac
done < "$PROMPT_FILE"

echo "Done!"
