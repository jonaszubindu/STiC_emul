#!/bin/bash
# Train the 4 ensemble members concurrently, one per GPU.
# Run on the GPU cluster from the repo root:
#   bash cluster/train_members.sh <label_dir> [<label_dir> ...] <ckpt_dir>
set -euo pipefail

ARGS=("$@")
CKPT="${ARGS[-1]}"
LABELS=("${ARGS[@]:0:$((${#ARGS[@]} - 1))}")
NGPU=${NGPU:-4}

echo "labels: ${LABELS[*]}  ->  ckpts: $CKPT  (${NGPU} GPUs)"
mkdir -p "$CKPT"
pids=()
for k in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$k python -m emulator.train_net "${LABELS[@]}" \
        --out "$CKPT" --member "$k" > "$CKPT/train_member_$k.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "all members done"
