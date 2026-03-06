#!/bin/bash
#SBATCH --job-name=bioclip2-coral-genus-ft-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=196G
#SBATCH --time=12:00:00
#SBATCH --output=/ibex/project/c2253/reefnet.ai/bioclip-2/logs/%j-%x.out
#SBATCH --error=/ibex/project/c2253/reefnet.ai/bioclip-2/logs/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yahia.battach@kaust.edu.sa

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
export OMP_NUM_THREADS=10

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/project/c2253/Xiang_Code/mae/pytoenv

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :; do
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

RUN_NAME=${RUN_NAME:-"coral_genus_ft_$(date +%Y%m%d_%H%M%S)"}
LOG_DIR=${LOG_DIR:-"./logs_finetune"}
CSV_PATH=/home/yahiab/reefnet_project/CoralNet_Images/patch_folders/reefnet_with_07_filteration_v02_hard_corals_no_exif_with_taxonomy.csv
CORAL_PATH_REPLACE_FROM=${CORAL_PATH_REPLACE_FROM:-"/ibex/project/c2253/CoralNet_Images/patch_folders/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}
CORAL_PATH_REPLACE_TO=${CORAL_PATH_REPLACE_TO:-"/raid/felembaa/reefnet_with_07_filteration_v02_hard_corals_no_exif/"}

torchrun \
  --nproc-per-node=${GPUS_PER_NODE} \
  --nnodes=${SLURM_JOB_NUM_NODES} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m src.training.main \
  --model hf-hub:imageomics/bioclip-2 \
  --train-data ${CSV_PATH} \
  --dataset-type coral_split \
  --coral-split-column split \
  --coral-train-split train \
  --coral-val-split val \
  --coral-target-level genus \
  --coral-caption-mode chain \
  --coral-path-key patch_path \
  --coral-path-replace-from "${CORAL_PATH_REPLACE_FROM}" \
  --coral-path-replace-to "${CORAL_PATH_REPLACE_TO}" \
  --coral-no-infer-genus-from-species \
  --coral-keep-missing-targets \
  --batch-size 128 \
  --epochs 100 \
  --lr 5e-4 \
  --warmup 500 \
  --wd 0.01 \
  --precision amp \
  --workers 16 \
  --logs ${LOG_DIR} \
  --name ${RUN_NAME}

echo "========================================="
echo "Job finished at : $(date)"
echo "========================================="
