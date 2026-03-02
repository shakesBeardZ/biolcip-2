We’re covered on both fronts now:

Genus-target run

slurm/finetune_coral_genus.sh launches a 30‑epoch job with --coral-target-level genus --coral-rank-only --coral-caption-mode rank_only, so the model optimises exactly what the open-domain metric scores. It still freezes the whole text tower—perfect for seeing how far genus-only tuning goes.
Species run with text projection unlocked

Added CLI knobs in src/training/params.py: --unlock-text-projection and --text-proj-lr.
Extended manual_lock_text_tower so you can thaw just the text projection while keeping the rest of the tower frozen; it re-enables gradients for that module even under DDP.
Updated the AdamW setup: if you pass --text-proj-lr, the text projection gets its own parameter group at that LR.
New script slurm/finetune_coral_species_proj.sh mirrors the original species run but adds --unlock-text-projection --text-proj-lr 5e-6 and limits training to 20 epochs for a quick check.
To launch:

# Genus-target experiment
sbatch slurm/finetune_coral_genus.sh

# Species + text projection experiment
sbatch slurm/finetune_coral_species_proj.sh
Both scripts auto-pick a fresh run name (timestamped) and log under ./logs_finetune. Once checkpoints land, re-run closed/open-domain evals (with the custom embedding bank) to see which direction gives the best lift.