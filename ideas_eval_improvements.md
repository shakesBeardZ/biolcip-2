**Evaluation Improvement Ideas (to try incrementally)**

This checklist captures experiments to bring closed‑set evaluation closer to the HF demo behavior and potentially improve accuracy on your coral data.

Baseline we have now
- Model: `--model hf-hub:imageomics/bioclip-2`
- Dataset: catalogue loader on your CSV
- Best current setting (genus): chain labels + OpenAI templates
  - `--data_loader catalogue --taxonomic_level genus --template_style openai` (no `--rank_only`)

Quick toggles (no code changes)
- [ ] Precision: compare AMP vs FP32
  - add `--precision fp32`
- [ ] Resolution: try larger eval size
  - add `--force-image-size 336` (revert if worse)
- [ ] Template styles on genus
  - `--template_style openai` (baseline), `bio`, `plain`
- [ ] Label scope on genus
  - compare chain (default) vs `--rank_only`

Prompt engineering (small code already in place)
- [ ] Rank‑only + templates (done): compare
  - `--rank_only --template_style plain|bio|openai`
- [ ] Prompt‑mix per class (code needed)
  - Idea: average text embeddings from multiple forms for each class: rank‑only (e.g., "Acropora"), chain‑to‑rank ("… Acroporidae Acropora"), and (for species) scientific binomial.
  - Flag: `--prompt_mix full` (to be implemented)

Domain templates for corals (code needed)
- [ ] Coral‑specific template set
  - Examples: "a stony coral of genus {}", "an underwater photo of {} coral", "a scleractinian coral: {}"
  - Flag: `--template_style coral` (to be implemented)

Test‑time augmentation (code needed)
- [ ] TTA (flips)
  - Average logits over original + horizontal flip
  - Flag: `--tta flips` (to be implemented)

Species fan‑out + rank aggregation (demo‑like)
- [ ] Predict at species, aggregate to target rank (genus/family)
  - Requires species text embeddings and names: `txt_emb_species.npy` and `txt_emb_species.json` (as in `bioclip2_demo/bioclip-2-demo/app.py`)
  - Steps:
    1. Place the files locally (no network fetch during eval).
    2. Add `--aggregate_to_rank` with args: `--species_emb_npy`, `--species_names_json`, and `--aggregate_rank genus|family|...`.
    3. Score images against all species embeddings (single pass per batch), softmax, then sum probabilities per target rank restricted to your closed set.
  - Benefit: mirrors demo’s open‑domain behavior and can help separate close taxa.

Common‑name enrichment (future)
- [ ] If common names become available, include them in class strings (e.g., "… Genus species (Common)") and evaluate effect.

Suggested run grid (genus)
- Baseline chain + openai:
  - `python -m src.evaluation.zero_shot_iid --model hf-hub:imageomics/bioclip-2 --data_loader catalogue --label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv --taxonomic_level genus --template_style openai --precision fp32 --batch-size 128 --workers 4 --logs ./logs/grid/genus_chain_openai`
- Chain + bio/plain:
  - swap `--template_style bio|plain`
- Rank‑only + templates:
  - add `--rank_only` and try `--template_style openai|bio|plain`
- Higher resolution:
  - add `--force-image-size 336` to the best above

Few‑shot (genus)
- Start with k=1 due to small classes:
  - `python -m src.evaluation.few_shot --model hf-hub:imageomics/bioclip-2 --data_loader catalogue --label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv --taxonomic_level genus --task_type all --kshot_list 1 --nfold 5 --batch-size 128 --workers 4 --logs ./logs/grid/genus_fewshot`
- Option (code change): filter underpopulated classes with a new flag (e.g., `--min_test_per_class 1`) to allow k>1 safely.

Priorities (my recommendation)
1) Lock in best zero‑shot: chain + openai + fp32; optionally 336.
2) Implement coral templates and/or prompt‑mix; re‑test.
3) If species assets can be staged locally, add aggregation mode for demo‑like behavior.
4) Add TTA if gains are needed.

