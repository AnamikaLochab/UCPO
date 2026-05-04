

set -euo pipefail
module purge
module load gcc/11.4.1

ENV=<>
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

echo "============================================"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-<unset>}"
echo "Start time: $(date)"
echo "============================================"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CHECKPOINT_BASE="../checkpoint_qwen_1e-6"
CHECKPOINT_STEP="global_step_318"
RESULTS_DIR="./results_step318_7b_qwen_1e-6"
METRICS_DIR="./metrics_step318_7b_qwen_1e-6"
N_SAMPLES=64

# Experiment names
EXP_NAMES=(
    "run8_iq_tau_only_0.2"
   
    
)

# ------------------------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------------------------
mkdir -p "$METRICS_DIR"
mkdir -p logs

# ------------------------------------------------------------------------------
# Run evaluation for each experiment
# ------------------------------------------------------------------------------
for EXP_NAME in "${EXP_NAMES[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${EXP_NAME}_s${N_SAMPLES}jsonl"
    OUTPUT_FILE="${METRICS_DIR}/${EXP_NAME}_metrics.json"
    MODEL_DIR="${CHECKPOINT_BASE}/q_7b_${EXP_NAME}/${CHECKPOINT_STEP}/actor/huggingface"
    echo "============================================"
    echo "Evaluating: ${EXP_NAME}"
    echo "Results: ${RESULT_FILE}"
    echo "Metrics: ${OUTPUT_FILE}"
    echo "Model: ${MODEL_DIR}"
    echo "Time: $(date)"
    echo "============================================"
    
    # Check if result file exists
    if [ ! -f "$RESULT_FILE" ]; then
        echo "WARNING: Result file not found: ${RESULT_FILE}"
        echo "Skipping ${EXP_NAME}..."
        echo ""
        continue
    fi
    
    python3 -m ucpo.main_eval \
        --result_file="$RESULT_FILE" \
        --output_file="$OUTPUT_FILE" \
        --model_name="$MODEL_DIR"
    
    echo "Completed: ${EXP_NAME}"
    echo ""
done

echo "============================================"
echo "All evaluations completed!"
echo "End time: $(date)"
echo "Metrics saved to: ${METRICS_DIR}"
echo "============================================"

