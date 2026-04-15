#!/bin/bash
# Usage: bash launch_sweep.sh <sweep_id>
# e.g.  bash launch_sweep.sh plt109/my-sbi-project/abc123

SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: bash launch_sweep.sh <sweep_id>"
    exit 1
fi

cd "$(dirname "$0")"

for GPU in 0 1 2 3 4 5; do
    echo "Launching agent on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU wandb agent $SWEEP_ID &
done

wait
echo "All agents done."
