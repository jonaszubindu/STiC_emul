#!/bin/bash
# Train the ensemble (one member per GPU) and predict the full map.
# Edit the lines marked EDIT, then from the repo root on the GPU cluster:
#   sbatch cluster/train_predict_gpu.slurm
#SBATCH --job-name=stic_emul_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # 1 GPU: the 4 networks train one after
                                      # another; gpu:4 trains them in parallel.
                                      # EDIT if your cluster wants e.g. --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5760M           # ~23 GB RAM; one training process
                                      # needs ~3-4 GB
#SBATCH --time=08:00:00
#SBATCH --partition=EDIT_GPU_PARTITION
#SBATCH --output=stic_emul_train_%j.log

set -eo pipefail

# EDIT: activate the environment where you ran  pip install -e '.[train]'
source ~/.bashrc
conda activate EDIT_ENV_NAME

set -u

# the work is on the GPU; keep CPU threads from oversubscribing
export OMP_NUM_THREADS=1

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

mkdir -p "$ROUND/ckpts"
bash cluster/train_members.sh "${LABELS[@]}" "$ROUND/ckpts"

python -m emulator.predict_map "$LAUNCH" --ckpt "$ROUND/ckpts" \
    --out "$ROUND/pred"
