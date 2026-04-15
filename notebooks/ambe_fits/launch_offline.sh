#!/bin/bash
# Offline fallback — runs all 9 combos directly without wandb sweep agent.
# Saves runs locally; sync later with: wandb sync wandb/

export WANDB_MODE=offline

cd "$(dirname "$0")"

BATCH_SIZES=(32 32 32 128 128 128 256 256 256)
LEARN_RATES=(0.0001 0.0003 0.001 0.0001 0.0003 0.001 0.0001 0.0003 0.001)

N_GPUS=6
N_RUNS=${#BATCH_SIZES[@]}

echo "Launching first 6 runs..."
for i in $(seq 0 5); do
    BS=${BATCH_SIZES[$i]}
    LR=${LEARN_RATES[$i]}
    echo "  GPU $i: bs=$BS lr=$LR"
    CUDA_VISIBLE_DEVICES=$i python train_histogram.py --batch_size $BS --learning_rate $LR &
done

wait
echo "First batch done. Launching remaining 3 runs..."

for i in $(seq 6 $((N_RUNS - 1))); do
    GPU=$((i - 6))
    BS=${BATCH_SIZES[$i]}
    LR=${LEARN_RATES[$i]}
    echo "  GPU $GPU: bs=$BS lr=$LR"
    CUDA_VISIBLE_DEVICES=$GPU python train_histogram.py --batch_size $BS --learning_rate $LR &
done

wait
echo "All done. To upload results: wandb sync wandb/"
