#!/bin/bash
# One bootstrap round, orchestrated from the machine that can reach both
# clusters over ssh (e.g. the laptop). Adapt HOST/PATH variables to your
# site.yaml values before first use.
#
#   bash cluster/round.sh <round_number>
#
# Round 0 special case: skip the tile stages and train directly on
# labels harvested from the existing (partially converged) map run:
#   ssh $CPU "cd $CPU_REPO && python -c \
#     \"from emulator import labels; labels.harvest_rundir('$MAP_RUN', apron=8)\""
set -euo pipefail

R=${1:?round number required}

# ---------------- adapt to your machines ----------------
CPU=cpu-cluster            # ssh alias, CPU cluster (SLURM + STiC.x)
GPU=gpu-cluster            # ssh alias, GPU cluster (4x H100)
CPU_REPO=/sml/zbindenj/STiC_emul
GPU_REPO=/scratch/zbindenj/STiC_emul
CPU_WORK=/sml/zbindenj/STiC_emul_runs
GPU_WORK=/scratch/zbindenj/STiC_emul_runs
STIC_BIN=/sml/zbindenj/coupled_stic/src/STiC.x
MAP_RUN=/sml/zbindenj/your_map_run_dir          # full-FOV coupled run dir
TEMPLATE=$MAP_RUN                               # tile extraction source
AUX="'/sml/zbindenj/coupled_stic/input_files', '/sml/zbindenj/coupled_stic'"
TILE=56; APRON=8; NTILES=32
# --------------------------------------------------------

RD=$CPU_WORK/round_$R
PREV=$CPU_WORK/round_$((R - 1))

echo "== round $R: prepare tiles on $CPU =="
INIT_OVERRIDE=""
if [ "$R" -gt 0 ]; then
    # initialize tiles from the previous round's predicted map
    INIT_OVERRIDE=", cfg_overrides={'max_inv_iter': '5', 'input_model': 'predicted_atmos.nc'}"
    scp "$PREV/predicted_atmos.nc" "$CPU:$TEMPLATE/predicted_atmos.nc"
else
    INIT_OVERRIDE=", cfg_overrides={'max_inv_iter': '5'}"
fi
ssh $CPU "cd $CPU_REPO && python - << PYEOF
from emulator import labels
sel = labels.select_tiles(['$TEMPLATE'], tile=$TILE, apron=$APRON,
                          n_tiles=$NTILES, seed=1000 + $R)
labels.prepare_runs(sel, '$RD', tile=$TILE, aux_search=($AUX,)$INIT_OVERRIDE)
labels.write_slurm('$RD', '$STIC_BIN', ntasks=32, time='04:00:00')
PYEOF"

echo "== round $R: submit + wait =="
JID=$(ssh $CPU "sbatch --parsable $RD/run_tiles.slurm")
echo "slurm job $JID"
while ssh $CPU "squeue -h -j $JID 2>/dev/null | grep -q ."; do
    sleep 120
done

echo "== round $R: harvest on $CPU =="
ssh $CPU "cd $CPU_REPO && python -c \
  \"from emulator import labels; labels.harvest('$RD', apron=$APRON)\""

echo "== round $R: labels -> $GPU, train, predict =="
rsync -az "$CPU:$RD/labels/" "/tmp/stic_labels_r$R/"
rsync -az "/tmp/stic_labels_r$R/" "$GPU:$GPU_WORK/round_$R/labels/"
ssh $GPU "cd $GPU_REPO && bash cluster/train_members.sh \
  $GPU_WORK/round_$R/labels $GPU_WORK/round_$R/ckpts"
# prediction needs the full-map obs files on the GPU side once:
#   rsync -az $CPU:$MAP_RUN/ $GPU:$GPU_WORK/map_run/   (first round only)
ssh $GPU "cd $GPU_REPO && python -m emulator.predict_map \
  $GPU_WORK/map_run --ckpt $GPU_WORK/round_$R/ckpts \
  --out $GPU_WORK/round_$R/pred"

echo "== round $R: prediction back to $CPU =="
rsync -az "$GPU:$GPU_WORK/round_$R/pred/" "/tmp/stic_pred_r$R/"
rsync -az "/tmp/stic_pred_r$R/" "$CPU:$RD/"
echo "round $R done: $RD/predicted_atmos.nc, nyquist.json"
