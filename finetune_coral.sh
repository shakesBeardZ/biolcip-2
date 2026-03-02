#!/bin/bash
#SBATCH --job-name=bioclip2-coral-ft-bf16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=196G
#SBATCH --time=23:30:00
#SBATCH --output=/ibex/project/c2253/reefnet.ai/bioclip-2/logs/%j-%x.out
#SBATCH --error=/ibex/project/c2253/reefnet.ai/bioclip-2/logs/%j-%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yahia.battach@kaust.edu.sa

# Environment setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=10
export SLURM_JOB_NUM_NODES=1

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/project/c2253/Xiang_Code/mae/pytoenv

# Dynamically pick an open port
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
    PORT=$(shuf -i $LOWERPORT-$UPPERPORT -n 1)
    ss -lpn | grep -q ":$PORT " || break
done

export MASTER_PORT=$PORT

echo "Running on $(hostname) with $GPUS_PER_NODE GPUs, port $PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Master address: $MASTER_ADDR"

echo "Available CUDA devices:"
nvidia-smi
module load cuda/11.8


RUN_NAME=${RUN_NAME:-"coral_species_ft_$(date +%Y%m%d_%H%M%S)"}
LOG_DIR=${LOG_DIR:-"./logs_finetune"}

torchrun \
  --nproc-per-node=${GPUS_PER_NODE:-4} \
  --nnodes=${SLURM_JOB_NUM_NODES:-1} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m src.training.main \
  --model hf-hub:imageomics/bioclip-2 \
  --train-data /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/inaturalist/inat_cotw_training_split_fixed_paths_cleaned_final.csv \
  --dataset-type coral_split \
  --coral-split-column split \
  --coral-train-split train \
  --coral-val-split val \
  --coral-target-level species \
  --coral-caption-mode chain \
  --batch-size 512 \
  --epochs 100 \
  --lr 5e-4 \
  --warmup 500 \
  --wd 0.01 \
  --precision bf16 \
  --lock-text \
  --lock-image \
  --lock-image-unlocked-groups 3 \
  --workers 10 \
  --logs ${LOG_DIR} \
  --name ${RUN_NAME}

echo "========================================="
echo "Job finished at    : $(date)"
echo "========================================="
