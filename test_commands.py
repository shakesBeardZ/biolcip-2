python - <<'PY'
import numpy as np, json, os
base="bioclip2_demo/bioclip-2-demo/components/data/species_embeddings"
E = np.load(os.path.join(base, "embeddings/txt_emb_species.npy"))
with open(os.path.join(base, "embeddings/txt_emb_species.json")) as f: names = json.load(f)
print("Emb shape:", E.shape, "names:", len(names))
names[i][0] is [Kingdom, Phylum, Class, Order, Family, Genus, Species]
print("Example:", " ".join(names[0][0]))
PY


python - <<'PY'
import json, os, pandas as pd
from dataset_catalogue import FranCatalogueLoader
names_path="bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json"
with open(names_path) as f: names=json.load(f)
def agg_label(tax, level="genus", rank_only=False):
    order={"kingdom":0,"phylum":1,"cls":2,"order":3,"family":4,"genus":5,"species":6}[level]
    if rank_only:
        return f"{tax[5]} {tax[6]}" if order==6 and len(tax)>=7 else tax[order]
    if order==6 and len(tax)>=7: return " ".join(tax[:6]+[f"{tax[5]} {tax[6]}"])
    return " ".join(tax[:order+1])

agg_set = set(agg_label(e[0], level="genus", rank_only=False) for e in names)
ds = FranCatalogueLoader("/home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/new_fran_catalogue.csv", taxonomic_level="genus", rank_only=False)
cls = set(ds.classes)
print("Covered classes:", len(cls & agg_set), "of", len(cls))
if len(cls-agg_set): print("Example missing:", list(cls-agg_set)[:5])
PY

python - <<'PY'
import json
from dataset_catalogue import FranCatalogueLoader
names_path="bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json"
with open(names_path) as f: names=json.load(f)
def agg_label(tax, level="genus"):
    order={"kingdom":0,"phylum":1,"cls":2,"order":3,"family":4,"genus":5,"species":6}[level]
    if order==6 and len(tax)>=7: return " ".join(tax[:6]+[f"{tax[5]} {tax[6]}"])
    return " ".join(tax[:order+1])
agg_set = set(agg_label(e[0], level="genus") for e in names)
ds = FranCatalogueLoader("/home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/new_fran_catalogue.csv", taxonomic_level="genus", normalize_anthozoa=True)
cls = set(ds.classes)
print("Covered classes:", len(cls & agg_set), "of", len(cls))
if len(cls-agg_set): print("Example missing:", list(cls-agg_set)[:5])
PY



python -m src.evaluation.zero_shot_iid \
--model hf-hub:imageomics/bioclip-2 \
--data_loader catalogue \
--label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv \
--taxonomic_level genus \
--rank_only \
--aggregate_to_rank \
--species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
--species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
--template_style openai \
--precision fp32 \
--batch-size 64  \
--workers 4 \
--logs ./logs/mimic_demo/genus_aggregate_rankonly \
--dump_predictions ./logs/mimic_demo/genus_aggregate_rankonly/pred_dump_top5.json \
--dump_topk 5



python -m src.evaluation.zero_shot_iid \
--model hf-hub:imageomics/bioclip-2 \
--data_loader catalogue \
--label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv \
--taxonomic_level species \
--rank_only \
--aggregate_to_rank \
--species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
--species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
--template_style openai \
--precision fp32 \
--batch-size 64  \
--workers 4 \
--logs ./logs/mimic_demo/species_aggregate_chain \
--dump_predictions ./logs/mimic_demo/species_aggregate_chain/pred_dump_top5.json \
--dump_topk 5


--------------------


python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--dump_misclassified_csv ./logs/mimic_demo/genus_open_domain_eval/misclassified.csv \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/fran_redsea_only_genera.csv \
--rank genus \
--normalize_anthozoa \
--species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
--species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
--batch-size 128 \
--workers 4 \
--precision fp32 \
--dump_predictions ./logs/mimic_demo/genus_open_domain_eval/pred_dump_top5.json \
--dump_topk 5 \
--filter_rank order \
--filter_value Scleractinia \
--restrict_hf_rank order \
--restrict_hf_value Scleractinia \
--hard_restrict_hf


python -m src.evaluation.closed_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
--csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/new_fran_catalogue.csv \
--rank genus \
--rank_only \
--template_style plain \
--normalize_anthozoa \
--batch-size 64  \
--workers 4  \
--filter_rank order \
--filter_value Scleractinia \
--precision fp32 \
--dump_predictions ./logs/closed_domain/genus_openai_pred.json \
--dump_misclassified_csv ./logs/closed_domain/genus_openai_misclassified.csv



python -m src.evaluation.open_domain_eval \
--model hf-hub:imageomics/bioclip-2 \
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



python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
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
