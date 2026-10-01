#!/bin/bash
# Train the ensemble and predict the full map — site settings for the
# UniBE GPU cluster (gpu-invest, H100 NVL). The generic template is
# train_predict_gpu.slurm. From the repo root on the GPU cluster:
#   sbatch cluster/train_predict_gpu_unibe.slurm
#
# With 1 GPU the ensemble members train one after another; for more GPUs
# raise the count in --gres (e.g. gpu:h100:4) and they train in parallel.
#SBATCH --job-name=stic_emul_train
#SBATCH --account=invest
#SBATCH --qos=job_gpu_kleint
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5760M
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-invest
#SBATCH --output=stic_emul_train_%j.log

set -eo pipefail

module load Python
module load Anaconda3
module load CUDA/12.6.0
eval "$(conda shell.bash hook)"
conda activate stic

set -u

# ignore packages in ~/.local: an old pip --user torch there (no H100
# support) shadows the env's torch
export PYTHONNOUSERSITE=1
# the work is on the GPU; keep CPU threads from oversubscribing
export OMP_NUM_THREADS=1
# write log lines immediately instead of in buffered chunks
export PYTHONUNBUFFERED=1

REPO=/storage/homefs/jz22g912/STiC_emul
LAUNCH=/storage/homefs/jz22g912/coupled_stic          # dir with input.cfg
ROUND=/storage/homefs/jz22g912/STiC_emul_runs/round0  # labels_* copied here

cd "$REPO"

# every labels_* dir that already holds records (map labels can be used
# before the tile labels exist)
LABELS=()
for d in "$ROUND"/labels_*; do
    if compgen -G "$d/*.npz" > /dev/null; then LABELS+=("$d"); fi
done
if [ ${#LABELS[@]} -eq 0 ]; then
    echo "no label records under $ROUND/labels_*"; exit 1
fi
echo "training on: ${LABELS[*]}"

# one output set per job, so a retraining never mixes with an earlier one
RUN_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
CKPT="$ROUND/ckpts_$RUN_ID"
PRED="$ROUND/pred_$RUN_ID"
CMP="$ROUND/compare_$RUN_ID"

bash cluster/train_members.sh "${LABELS[@]}" "$CKPT"

python -m emulator.predict_map "$LAUNCH" --ckpt "$CKPT" --out "$PRED"

# compare with the inversion if full-map labels exist
if [ -f "$ROUND/labels_map/full_map.npz" ]; then
    python -m emulator.compare_maps --labels "$ROUND/labels_map/full_map.npz" \
        --pred "$PRED" --ckpt "$CKPT" --out "$CMP"
fi
echo "checkpoints: $CKPT"
echo "prediction:  $PRED"
echo "comparison:  $CMP"
