#!/bin/bash
#SBATCH --job-name=eval-bioclip2
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=08:30:00
#SBATCH --output=/ibex/project/c2253/reefnet.ai/bioclip-2/logs/%j-%x.out
#SBATCH --error=/ibex/project/c2253/reefnet.ai/bioclip-2/logs/%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yahia.battach@kaust.edu.sa

# --- Environment -----------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
export SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
export OMP_NUM_THREADS=8

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/project/c2253/yahia_code/envs/pytoenv

# Pick an open rendezvous port
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :; do
  PORT=$(shuf -i $LOWERPORT-$UPPERPORT -n 1)
  ss -lpn | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT

echo "Running on $(hostname) with $GPUS_PER_NODE GPUs, port $PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Master address: $MASTER_ADDR"

# --- Paths & hyperparameters ----------------------------------------------
RUN_NAME=${RUN_NAME:-"coral_scratch_vitl_$(date +%Y%m%d_%H%M%S)"}
LOG_DIR=${LOG_DIR:-"./logs_scratch"}
CSV_PATH=/ibex/project/c2253/yahia_code/data_preprocessing_scripts/reefnet.ai/reefnet_scleractinia_inat_cotw_combined_final.csv

# --- Training --------------------------------------------------------------
python -m tools.eval_checkpoints \
  --checkpoints-dir /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/ \
  --eval-open --eval-closed \
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rsg_corals_only \
  --rank genus \
  --normalize_anthozoa \
  --template_style plain \
  --workers 8 \
  --batch-size 256 \
  --species_emb_npy custom_embeddings/coral_genus_chain_finetune_all_data_v2_emb.npy \
  --species_names_json custom_embeddings/coral_genus_chain_finetune_all_data_v2_names.json \
  --output-csv logs/eval/coral_genus_chain_finetune_all_data_v2_names_eval.csv

# --- Footer ---------------------------------------------------------------
echo "========================================="
echo "Job finished at : $(date)"
echo "========================================="
