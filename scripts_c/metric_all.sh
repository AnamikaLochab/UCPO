

set -euo pipefail
module purge
module load gcc/11.4.1

ENV=<>
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

echo "host=$(hostname)"
echo "CVD=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi -L || true


python3 -m ucpo.main_metric