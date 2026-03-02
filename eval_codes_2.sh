python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_ft_20250926_125110/checkpoints/epoch_30.pt \
--dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval/misclassified.csv \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
--normalize_anthozoa \
--rank genus \
--rank_only \
--species_emb_npy custom_embeddings/coral_genus_rank_emb.npy \
--species_names_json custom_embeddings/coral_genus_rank_names.json \
--batch-size 128 \
--workers 4 \
--precision amp \
--dump_predictions ./logs/mimic_demo/species_open_domain_eval/pred_dump_top5.json \
--dump_topk 5 \
--filter_rank order \
--filter_value Scleractinia \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf

both of them:
top1: 56.45
top3: 79.28
top5: 87.32



python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_ft_20250926_035254/checkpoints/epoch_100.pt \
--dataset_loader rsg \
--csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
--rank genus \
--rank_only \
--normalize_anthozoa \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf \
--species_emb_npy custom_embeddings/coral_species_chain_emb.npy \
--species_names_json custom_embeddings/coral_species_chain_names.json \
--batch-size 128 \
--workers 4 \
--dump_predictions ./logs/rsg_demo/pred_dump_top5.json \
--dump_topk 5 \
--dump_misclassified_csv ./logs/rsg_demo/misclassified.csv \
--rsg_corals_only


torchrun \
  --nproc-per-node=2 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  -m src.training.main \
  --model hf-hub:imageomics/bioclip-2 \
  --train-data /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/inaturalist/inat_cotw_training_split_fixed_paths_cleaned_final.csv \
  --dataset-type coral_split \
  --coral-split-column split \
  --coral-train-split train \
  --coral-val-split val \
  --coral-target-level genus \
  --coral-rank-only \
  --coral-caption-mode rank_only \
  --batch-size 128 \
  --epochs 30 \
  --lr 5e-5 \
  --warmup 500 \
  --wd 0.01 \
  --precision amp \
  --lock-text \
  --lock-image \
  --lock-image-unlocked-groups 3 \
  --workers 16 \
  --logs ./logs_finetune \
  --name coral_genus_ft_20241030_2node




python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_ft_20250926_125110/checkpoints/epoch_30.pt \
  --dataset_loader csv \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank genus \
  --rank_only \
  --normalize_anthozoa \
  --species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
  --species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
  --batch-size 128 \
  --workers 4 \
  --precision fp32 \
  --filter_rank order \
  --filter_value Scleractinia \
  --restrict_hf_rank order \
  --restrict_hf_value Scleractinia \
  --hard_restrict_hf \
  --dump_predictions ./logs/mimic_demo/genus_open_domain_eval_genus_run/pred_dump_top5.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/mimic_demo/genus_open_domain_eval_genus_run/misclassified.csv



python -m tools.export_coral_embeddings \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/inaturalist/inat_cotw_training_split_fixed_paths_cleaned_final.csv\
  --split-column split --include-splits train,val \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_genus_ft_20250926_125110/checkpoints/epoch_30.pt \
  --caption-mode rank_only \
  --normalize-anthozoa \
  --output-prefix ./custom_embeddings/coral_genus_rank



--- 
python -m tools.export_coral_embeddings \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/inaturalist/inat_cotw_training_split_fixed_paths_cleaned_final.csv \
  --split-column split --include-splits train,val \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250926_195759/checkpoints/epoch_30.pt\
  --caption-mode chain \
  --normalize-anthozoa \
  --output-prefix ./custom_embeddings/coral_genus_chain



python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_ft_20250926_125110/checkpoints/epoch_30.pt \
--dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval/misclassified.csv \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
--normalize_anthozoa \
--rank genus \
--rank_only \
--species_emb_npy custom_embeddings/coral_genus_rank_emb.npy \
--species_names_json custom_embeddings/coral_genus_rank_names.json \
--batch-size 128 \
--workers 4 \
--precision amp \
--dump_predictions ./logs/mimic_demo/species_open_domain_eval/pred_dump_top5.json \
--dump_topk 5 \
--filter_rank order \
--filter_value Scleractinia \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf

--- closed domain: 
python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250926_195835/checkpoints/epoch_100.pt \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank genus \
  --normalize_anthozoa \
  --template_style plain \
  --batch-size 128 --workers 4 --precision amp \
  --dump_predictions ./logs/closed_domain/genus/pred_dump.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/closed_domain/genus/misclassified.csv



-- closed domain species level: 
python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_ft_20250926_022142/checkpoints/epoch_100.pt \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank genus \
  --normalize_anthozoa \
  --template_style plain \
  --batch-size 128 --workers 4 --precision amp \
  --dump_predictions ./logs/closed_domain/species/pred_dump.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/closed_domain/species/misclassified.csv

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


----- project finetuning script

python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_140224/checkpoints/epoch_27.pt \
--dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval/misclassified.csv \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
--normalize_anthozoa \
--rank genus \
--species_emb_npy custom_embeddings/coral_genus_chain_emb.npy \
--species_names_json custom_embeddings/coral_genus_chain_names.json \
--batch-size 128 \
--workers 4 \
--precision amp \
--dump_predictions ./logs/mimic_demo/species_open_domain_eval/pred_dump_top5.json \
--dump_topk 5 \
--filter_rank order \
--filter_value Scleractinia \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf

python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_140224/checkpoints/epoch_10.pt \
--dataset_loader rsg \
--csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
--rank genus \
--normalize_anthozoa \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf \
--species_emb_npy custom_embeddings/coral_genus_chain_emb.npy \
--species_names_json custom_embeddings/coral_genus_chain_names.json \
--batch-size 128 \
--workers 4 \
--dump_predictions ./logs/rsg_demo/pred_dump_top5.json \
--dump_topk 5 \
--dump_misclassified_csv ./logs/rsg_demo/misclassified.csv \
--rsg_corals_only

python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_140224/checkpoints/epoch_27.pt \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank genus \
  --normalize_anthozoa \
  --template_style plain \
  --batch-size 128 --workers 4 --precision amp \
  --dump_predictions ./logs/closed_domain/genus/pred_dump.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/closed_domain/genus/misclassified.csv

python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_species_proj_ft_20250927_140224/checkpoints/epoch_27.pt \
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rank genus \
  --normalize_anthozoa \
  --rsg_corals_only \
  --batch-size 128 --workers 4 --precision amp \
  --template_style plain

--- models trained on reefnet data 

python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/epoch_47.pt \
--dump_misclassified_csv ./logs/mimic_demo/species_open_domain_eval/misclassified.csv \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
--normalize_anthozoa \
--rank genus \
--species_emb_npy custom_embeddings/coral_genus_chain_emb.npy \
--species_names_json custom_embeddings/coral_genus_chain_names.json \
--batch-size 128 \
--workers 4 \
--precision amp \
--dump_predictions ./logs/mimic_demo/species_open_domain_eval/pred_dump_top5.json \
--dump_topk 5 \
--filter_rank order \
--filter_value Scleractinia \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf

python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/epoch_47.pt \
--dataset_loader rsg \
--csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
--rank genus \
--normalize_anthozoa \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf \
--species_emb_npy custom_embeddings/coral_genus_chain_emb.npy \
--species_names_json custom_embeddings/coral_genus_chain_names.json \
--batch-size 128 \
--workers 4 \
--dump_predictions ./logs/rsg_demo/pred_dump_top5.json \
--dump_topk 5 \
--dump_misclassified_csv ./logs/rsg_demo/misclassified.csv \
--rsg_corals_only


python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/epoch_51.pt \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank genus \
  --normalize_anthozoa \
  --template_style plain \
  --batch-size 128 --workers 4 --precision amp \
  --dump_predictions ./logs/closed_domain/genus/pred_dump.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/closed_domain/genus/misclassified.csv


python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/epoch_50.pt \
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rank genus \
  --normalize_anthozoa \
  --rsg_corals_only \
  --batch-size 128 --workers 4 --precision amp \
  --template_style plain

----- new checkpoint for the all the data finetuning
-- first export embeddings

python -m tools.export_coral_embeddings \
  --csv /ibex/project/c2253/yahia_code/data_preprocessing_scripts/reefnet.ai/reefnet_scleractinia_inat_cotw_combined_final.csv \
  --split-column split \
  --include-splits train,val \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/epoch_100.pt \
  --caption-mode chain \
  --normalize-anthozoa \
  --output-prefix ./custom_embeddings/coral_genus_chain_finetune_all_data_v2

python -m tools.eval_checkpoints \
  --checkpoints-dir /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/ \
  --eval-open --eval-closed \
  --dataset_loader catalogue \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv \
  --rank genus \
  --normalize_anthozoa \
  --template_style plain \
  --species_emb_npy custom_embeddings/coral_genus_chain_finetune_all_data_emb.npy\
  --species_names_json custom_embeddings/coral_genus_chain_finetune_all_data_names.json \
  --output-csv logs/checkpoint_eval.csv


python -m tools.eval_checkpoints   --checkpoints-dir /ibex/project/c2253/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/   --eval-open --eval-closed   --dataset_loader catalogue   --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera_cleaned.csv   --rank genus   --normalize_anthozoa   --template_style plain   --workers 4   --batch-size 128   --species_emb_npy custom_embeddings/coral_genus_chain_finetune_all_data_v2_emb.npy   --species_names_json custom_embeddings/coral_genus_chain_finetune_all_data_v2_names.json   --output-csv logs/eval/coral_genus_chain_finetune_all_data_v2_names_eval.csv


--- eval from scratch
// generate embeddings

python -m tools.export_coral_embeddings \
  --csv /ibex/project/c2253/yahia_code/data_preprocessing_scripts/reefnet.ai/reefnet_scleractinia_inat_cotw_combined_final.csv \
  --split-column split \
  --include-splits train,val \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_scratch/coral_scratch_vitl_20250929_052639/checkpoints/epoch_100.pt \
  --target-level genus \
  --caption-level genus \
  --caption-mode chain \
  --apply-normalize-anthozoa \
  --output-prefix ./custom_embeddings/coral_genus_chain_scratch_all_data_v2

// closed domain eval 


python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_genus_chain_ft_20250927_230843/checkpoints/epoch_100.pt  \
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rank genus \
  --normalize_anthozoa \
  --rsg_corals_only \
  --batch-size 128 --workers 4 --precision amp \
  --template_style plain


python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_genus_chain_ft_20250926_195835/checkpoints/epoch_100.pt\
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rank genus \
  --normalize_anthozoa \
  --rsg_corals_only \
  --class_list redsea_genera.txt \
  --batch-size 128 --workers 4 --precision amp \
  --template_style plain


// open domain eval

python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--checkpoint logs_scratch/coral_scratch_vitl_20250929_052639/checkpoints/epoch_100.pt \
--dataset_loader rsg \
--csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
--rank genus \
--normalize_anthozoa \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf \
--species_emb_npy custom_embeddings/coral_genus_chain_scratch_all_data_v2_emb.npy \
--species_names_json custom_embeddings/coral_genus_chain_scratch_all_data_v2_names.json \
--batch-size 128 \
--workers 4 \
--dump_predictions ./logs/rsg_demo/pred_dump_top5.json \
--dump_topk 5 \
--dump_misclassified_csv ./logs/rsg_demo/misclassified.csv \
--rsg_corals_only



python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --checkpoint logs_finetune/coral_genus_chain_ft_20250927_230843/checkpoints/epoch_100.pt  \
  --dataset_loader rsg \
  --csv /ibex/project/c2253/RSG_test_data/TRSP-SEZ_exposed/rsg_ann_with_patches_512_0_17870_no_Diploastrea.csv \
  --rank genus \
  --normalize_anthozoa \
  --restrict_hf_rank order \
  --restrict_hf_value Scleractinia \
  --hard_restrict_hf \
  --class_list redsea_genera.txt \
  --species_emb_npy custom_embeddings/coral_genus_chain_finetune_all_data_v2_emb.npy \
  --species_names_json custom_embeddings/coral_genus_chain_finetune_all_data_v2_names.json \
  --batch-size 128 \
  --workers 4 \
  --dump_predictions ./logs/rsg_demo/pred_dump_top5.json \
  --dump_topk 5 \
  --dump_misclassified_csv ./logs/rsg_demo/misclassified.csv \
  --rsg_corals_only


/home/yahiab/reefnet_project/reefnet.ai/bioclip-2/logs_finetune/coral_genus_chain_ft_20250928_010055/checkpoints/epoch_100.pt

