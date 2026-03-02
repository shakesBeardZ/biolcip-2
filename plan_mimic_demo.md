**Goal**

- Match or exceed the HF Gradio demo’s zero-shot behavior on your coral data, then stage fine-tuning on your separate coral dataset.

**What The Demo Does (bioclip2_demo/bioclip-2-demo/app.py)**

- Loads model via `create_model("hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True)` on CPU; compiles with `torch.compile(model)`.
- Preprocess: plain torchvision pipeline (Resize to 224, OpenAI mean/std). Equivalent to our model’s eval preprocess.
- Zero-shot (user-provided classes):
  - Uses OpenAI ImageNet-style templates on the provided class strings.
  - Encodes text prompts per class, normalizes, averages, cosine similarity with normalized image features, scales by `logit_scale.exp()`, returns softmax probs.
- Open-domain mode (entire TreeOfLife-200M):
  - Precomputed text embeddings for all species (`txt_emb_species.npy`) + taxonomy names (`txt_emb_species.json`).
  - For ranks above species, sums species-level probabilities mapped to each higher-rank taxon; reports top-k.

**What We Observed Locally**

- Zero-shot on genus with long chain labels and OpenAI templates: OK, but below demo expectations.
- Switching to scientific-name only decreased accuracy on your set.
- Dumped per-image predictions show mismatches; likely due to prompt style and label construction.

**Key Alignment Levers**

- Label strings: rank-only (e.g., just `Genus` or `Genus species`) generally embed better than long taxonomic chains.
- Prompt templates: demo uses OpenAI ImageNet templates but on short names. For long chains, plain prompts often work better; for short names, either OpenAI or plain can work.
- Precision & transforms: keep eval transforms identical to model preprocess; consider `--precision fp32` to remove AMP variance for small evals.

**Plan (Phased)**

1) Align Zero-Shot To Demo
- Use `--model hf-hub:imageomics/bioclip-2` (no `--pretrained`).
- Switch to rank-only labels from the catalogue: `--rank_only`.
- Try `--template_style plain` and `--template_style openai` and keep the best.
- For genus: `--taxonomic_level genus --rank_only` ⇒ labels like "Acropora".
- For species: `--taxonomic_level species --require_species --rank_only` ⇒ labels like "Acropora natalensis".
- Evaluate and dump per-image predictions for manual inspection.

2) Compare Against Demo Behavior
- For a few images, replicate the demo’s zero-shot input: short class lists (3–5 options), scientific and/or common names when available.
- Verify preprocessing parity (our pipeline already matches OpenAI mean/std; image size equals chosen model’s preprocess size).
- If needed, try template variants: `--template_style bio` (adds few paraphrases).

3) Stabilize Few-Shot
- Start with `kshot=1` to accommodate small classes; increase only if per-class counts support it.
- Optionally: patch few_shot.py to filter out underpopulated classes (skip classes with `< kshot + min_test`), controlled by a flag; this prevents assertion failures and reduces noise.

4) Document a “Demo-Equivalent” Eval Preset
- Save exact working CLI (model, taxonomy level, rank_only, template_style, precision, batch-size) that gives the strongest results on your set.
- Keep a short script to re-run and dump predictions.

5) Prepare Fine-Tuning Plan (Outline)
- Data: convert your coral training data to the webdataset or file-list format expected by `src/training/main.py`, or adapt `src/training/data.py` to your CSV format.
- Strategy: start with vision-head fine-tuning while freezing (or low lr) for the text tower, leverage BioCLIP-2 visual projector for biology; consider class-balanced sampling.
- Config: small LR (1e-5–5e-5), cosine schedule with warmup, mixed precision, early stopping on validation.
- Checkpoints: save best by validation metric; export to a path that `create_model_and_transforms` can load via `--pretrained` (local path).

**Concrete Commands (To Mimic Demo on Your Data)**

- Zero-shot genus (rank-only, plain prompts, dump top-5 for inspection):
  - `python -m src.evaluation.zero_shot_iid \
    --model hf-hub:imageomics/bioclip-2 \
    --data_loader catalogue \
    --label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv \
    --taxonomic_level genus \
    --rank_only \
    --template_style plain \
    --precision fp32 \
    --batch-size 128 --workers 4 \
    --logs ./logs/mimic_demo/genus_plain_rankonly \
    --dump_predictions ./logs/mimic_demo/genus_plain_rankonly/pred_dump_top5.json \
    --dump_topk 5`

- Zero-shot species (strict, rank-only, plain):
  - `python -m src.evaluation.zero_shot_iid \
    --model hf-hub:imageomics/bioclip-2 \
    --data_loader catalogue \
    --label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv \
    --taxonomic_level species \
    --require_species \
    --rank_only \
    --template_style plain \
    --precision fp32 \
    --batch-size 128 --workers 4 \
    --logs ./logs/mimic_demo/species_plain_rankonly \
    --dump_predictions ./logs/mimic_demo/species_plain_rankonly/pred_dump_top5.json \
    --dump_topk 5`

- Few-shot genus (1-shot due to small classes):
  - `python -m src.evaluation.few_shot \
    --model hf-hub:imageomics/bioclip-2 \
    --data_loader catalogue \
    --label_filename /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv \
    --taxonomic_level genus \
    --rank_only \
    --task_type all \
    --kshot_list 1 \
    --nfold 5 \
    --batch-size 128 --workers 4 \
    --logs ./logs/mimic_demo/genus_fewshot_rankonly`

If plain prompts underperform, repeat zero-shot with `--template_style openai` or `--template_style bio` and pick best.

**Validation Steps**

- Review `pred_dump_top5.json` for representative images; optionally reduce to CSV (top-1) for quick scans.
- Check per-class accuracies to identify weak taxa; consider merging sparse classes or using family-level for those in zero-shot.
- Confirm that class counts support k-shot before increasing k.

**Fine-Tuning Preparation (Next Phase)**

- Data ingestion: decide between adapting `src/training/data.py` to your CSV or writing a small converter to the existing image/label format.
- Loss & heads: keep BioCLIP-2’s dual projector; start with vision encoder unfrozen (low LR) and text encoder frozen or at very low LR.
- Monitoring: use a held-out validation set; periodically run zero-shot on your evaluation CSV to ensure generalization.

**Open Questions To Resolve Before Finetune**

- Which taxonomy level to optimize (genus vs species)?
- How many examples per class (to set batch size, k-shot baselines, and sampling strategy)?
- Do we need common-name prompts or bilingual prompts for your data?

