#!/bin/bash
# Train the ensemble members on whatever GPUs the job was given.
# Members are distributed round-robin over the GPUs in
# CUDA_VISIBLE_DEVICES (set by SLURM); members on the same GPU run one
# after another. 1 GPU: all members sequentially. 4 GPUs: all at once.
#
#   bash cluster/train_members.sh <label_dir> [<label_dir> ...] <ckpt_dir>
#
# Environment:
#   NMEM        ensemble size (default 4)
#   TRAIN_ARGS  extra train_net arguments, e.g. "--steps 200" for a test
set -euo pipefail

ARGS=("$@")
CKPT="${ARGS[${#ARGS[@]} - 1]}"
LABELS=("${ARGS[@]:0:$((${#ARGS[@]} - 1))}")
NMEM=${NMEM:-4}
read -ra EXTRA <<< "${TRAIN_ARGS:-}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPUS=(0)
fi
NG=${#GPUS[@]}

echo "labels: ${LABELS[*]}  ->  ckpts: $CKPT"
echo "members: $NMEM  on GPUs: ${GPUS[*]}"
mkdir -p "$CKPT"

pids=()
for i in $(seq 0 $((NG - 1))); do
    (
        for ((k = i; k < NMEM; k += NG)); do
            echo "member $k -> GPU ${GPUS[$i]}"
            CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -m emulator.train_net \
                "${LABELS[@]}" --out "$CKPT" --member "$k" --members "$NMEM" \
                ${EXTRA[@]+"${EXTRA[@]}"} > "$CKPT/train_member_$k.log" 2>&1
        done
    ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "all members done"
