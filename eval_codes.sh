python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint checkpoints/coral_species/best.pt \
  --dataset_loader csv \
  --csv /path/to/coral_val.csv \
  --rank genus --rank_only --normalize_anthozoa \
  --restrict_hf_rank order --restrict_hf_value Scleractinia --hard_restrict_hf \
  --species_emb_npy bioclip2_demo/.../txt_emb_species.npy \
  --species_names_json bioclip2_demo/.../txt_emb_species.json \
  --batch-size 128 --workers 8 --precision amp



python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_ft_20250926_035254/checkpoints/epoch_100.pt \
--dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval/misclassified.csv \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
--rank species \
--rank_only \
--require_species \
--normalize_anthozoa \
--species_emb_npy custom_embeddings/coral_species_chain_emb.npy \
--species_names_json custom_embeddings/coral_species_chain_names.json \
--batch-size 128 \
--workers 4 \
--precision fp32 \
--dump_predictions ./logs/mimic_demo/species_open_domain_eval/pred_dump_top5.json \
--dump_topk 5 \
--filter_rank order \
--filter_value Scleractinia \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf


python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_ft_20250926_012307/checkpoints/epoch_29.pt \
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rank genus \
  --rank_only \
  --normalize_anthozoa \
  --restrict_hf_rank order \
  --restrict_hf_value Scleractinia \
  --hard_restrict_hf \
  --species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
  --species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
  --batch-size 128 \
  --workers 4 \
  --precision fp32 \
  --dump_predictions ./logs/rsg_demo/pred_dump_top5.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/rsg_demo/misclassified.csv \
  --rsg_corals_only


python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_ft_20250926_012307/checkpoints/epoch_29.pt \
--dataset_loader acropora \
--acropora_split val \
--rank genus \
--rank_only \
--normalize_anthozoa \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf \
--species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
--species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
--batch-size 128 \
--workers 4 \
--precision fp32 \
--dump_predictions ./logs/acropora_demo/pred_dump_top5.json \
--dump_topk 5 \
--dump_misclassified_csv ./logs/acropora_demo/misclassified.csv


python -m tools.export_coral_embeddings \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/inaturalist/inat_cotw_training_split_fixed_paths_cleaned_final.csv \
  --split-column split \
  --include-splits train,val \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_species_ft_20250926_035254/checkpoints/epoch_100.pt \
  --caption-mode chain \
  --normalize-anthozoa \
  --output-prefix ./custom_embeddings/coral_species_chain