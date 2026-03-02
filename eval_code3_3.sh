python -m tools.export_coral_embeddings \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/inaturalist/inat_cotw_training_split_fixed_paths_cleaned_final.csv \
  --split-column split --include-splits train,val \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_species_ft_20250926_012307/checkpoints/epoch_100.pt \
  --target-level species \
  --caption-level species \
  --caption-mode chain \
  --apply-normalize-anthozoa \
  --output-prefix ./custom_embeddings/coral_species_ft_20250926_012307

Wrote embeddings: custom_embeddings/coral_species_ft_20250926_012307_emb.npy
Wrote names:      custom_embeddings/coral_species_ft_20250926_012307_names.json



python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_135933/checkpoints/epoch_100.pt \
  --dataset_loader csv \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank species \
  --require_species \
  --filter_rank genus \
  --filter_value Acropora \
  --normalize_anthozoa \
  --species_emb_npy custom_embeddings/coral_species_ft_20250926_035254_emb.npy \
  --species_names_json custom_embeddings/coral_species_ft_20250926_035254_names.json \
  --batch-size 128 \
  --workers 4 \
  --precision amp \
  --filter_rank order \
  --filter_value Scleractinia \
  --restrict_hf_rank order \
  --restrict_hf_value Scleractinia \
  --hard_restrict_hf \
  --dump_predictions ./logs/mimic_demo/species_open_domain_eval_acropora/pred_dump_top5.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval_acropora/misclassified.csv



python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_135933/checkpoints/epoch_100.pt \
  --dataset_loader csv \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank species \
  --rank_only \
  --require_species \
  --filter_rank genus \
  --filter_value Acropora \
  --normalize_anthozoa \
  --species_emb_npy custom_embeddings/coral_species_chain_proj_ft_1_emb.npy \
  --species_names_json custom_embeddings/coral_species_chain_proj_ft_1_names.json \
  --batch-size 128 \
  --workers 4 \
  --precision amp \
  --restrict_hf_rank order \
  --restrict_hf_value Scleractinia \
  --hard_restrict_hf \
  --dump_predictions ./logs/mimic_demo/species_open_domain_eval_acropora/pred_dump_top5.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval_acropora/misclassified.csv



python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_species_ft_20250926_012307/checkpoints/epoch_100.pt \
  --dataset_loader csv \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank species \
  --rank_only \
  --require_species \
  --filter_rank genus \
  --filter_value Acropora \
  --normalize_anthozoa \
  --species_emb_npy custom_embeddings/coral_species_ft_20250926_012307_emb.npy \
  --species_names_json custom_embeddings/coral_species_ft_20250926_012307_names.json \
  --batch-size 128 \
  --workers 4 \
  --precision amp \
  --restrict_hf_rank order \
  --restrict_hf_value Scleractinia \
  --hard_restrict_hf \
  --dump_predictions ./logs/mimic_demo/species_open_domain_eval_acropora/pred_dump_top5.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval_acropora/misclassified.csv


python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_135933/checkpoints/epoch_100.pt \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank species \
  --require_species \
  --filter_rank genus \
  --filter_value Acropora \
  --normalize_anthozoa \
  --template_style plain \
  --batch-size 128 --workers 4 --precision amp \
  --dump_predictions ./logs/closed_domain/species/pred_dump.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/closed_domain/species/misclassified.csv
