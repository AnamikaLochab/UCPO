
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
nvidia-smi -L || true

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CHECKPOINT_BASE="../checkpoint_re"
CHECKPOINT_STEP="global_step_318"
TEST_DATADIR="./dataset/test.json"
OUTPUT_DIR="./results_step318"
PROMPT_KEY="prompt"

# Inference parameters
TEMPERATURE=1
MAX_TOKENS=3072
N_SAMPLES=64
TENSOR_PARALLEL=1
GPU_MEM_UTIL=0.95
EXP_NAMES=(
    
    "run8_iq_tau_only_0.2"
)

# ------------------------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ------------------------------------------------------------------------------
# Run inference for each experiment
# ------------------------------------------------------------------------------
for EXP_NAME in "${EXP_NAMES[@]}"; do
    MODEL_DIR="${CHECKPOINT_BASE}/ds_1.5b_${EXP_NAME}/${CHECKPOINT_STEP}/actor/huggingface"
    OUTPUT_FILE="${OUTPUT_DIR}/${EXP_NAME}_s${N_SAMPLES}.jsonl"
    
    echo "============================================"
    echo "Running inference for: ${EXP_NAME}"
    echo "Model: ${MODEL_DIR}"
    echo "Output: ${OUTPUT_FILE}"
    echo "Time: $(date)"
    echo "============================================"
    
    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "WARNING: Model directory not found: ${MODEL_DIR}"
        echo "Skipping ${EXP_NAME}..."
        continue
    fi
    
    python3 -m ucpo.main_infer \
        --model="$MODEL_DIR" \
        --input_file="$TEST_DATADIR" \
        --output_file="$OUTPUT_FILE" \
        --prompt_key="$PROMPT_KEY" \
        --tensor_parallel_size="$TENSOR_PARALLEL" \
        --gpu_memory_utilization="$GPU_MEM_UTIL" \
        --temperature="$TEMPERATURE" \
        --max_tokens="$MAX_TOKENS" \
        --n_samples="$N_SAMPLES"
    
    echo "Completed: ${EXP_NAME}"
    echo ""
done

echo "============================================"
echo "All inference jobs completed!"
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"
