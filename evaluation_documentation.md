**Purpose**

This note documents how we evaluate BioCLIP‑2 on your coral data in two modes, mirroring the Hugging Face demo (open‑domain) and using only your label set (closed‑domain). It captures flags, workflows, and known good settings so we can pick up quickly next time.

**Data & Loader**

- CSV format: `Path` (image path; absolute or relative to `--data_root`) and taxonomy columns: `kingdom, phylum, class/cls, order, family, genus, species`.
- Loader: `dataset_catalogue.FranCatalogueLoader`
  - `taxonomic_level`: which rank to use for labels (e.g., `genus`, `species`).
  - `rank_only` (loader arg): if set, the label is just the chosen rank token (e.g., `Acropora`). Otherwise the label is the full chain‑to‑rank (e.g., `Animalia Cnidaria Anthozoa Scleractinia … Acropora`).
  - `normalize_anthozoa` (loader arg): map `class=Hexacorallia` under `phylum=Cnidaria` to `Anthozoa` to align with TreeOfLife taxonomy.

**Open‑Domain (Demo‑Like)**

Script: `src/evaluation/open_domain_eval.py`

- What it does:
  - Loads BioCLIP‑2 + preprocess.
  - Loads species text embeddings + names (HF files): `txt_emb_species.npy` and `txt_emb_species.json`.
  - For each image: encodes image → scores against (all or restricted) species embeddings → softmax → aggregates species probabilities to your target rank (e.g., genus) → evaluates top‑k against your CSV labels.
- Key flags:
  - `--rank genus|…`: aggregation level for scoring and metrics.
  - `--rank_only`: improves label matching (uses just the rank token); does not change scoring.
  - `--normalize_anthozoa`: normalize your CSV chain to match HF taxonomy.
  - `--filter_rank / --filter_value`: optionally filter your dataset rows (e.g., only `order=Scleractinia`).
  - `--restrict_hf_rank / --restrict_hf_value`: restrict the HF open‑domain species pool by taxonomy (e.g., `order=Scleractinia`).
  - `--hard_restrict_hf`: physically slices the HF species embeddings and names to the restricted pool before scoring (true “image · species_emb(of Scleractinia)”).
  - Dumps:
    - `--dump_predictions`: JSON per‑image top‑k (labels with probabilities).
    - `--dump_misclassified_csv`: CSV of misclassified examples; contains `path`, `pred_genus_chain` (chain‑to‑genus) and `pred_species_chain` (full chain, demo‑style, with common name when available).
- Typical genus command (hard‑restricted Scleractinia):
```
python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --csv /path/to/val.csv \
  --rank genus --rank_only --normalize_anthozoa \
  --restrict_hf_rank order --restrict_hf_value Scleractinia --hard_restrict_hf \
  --species_emb_npy data/species_embeddings_scleractinia/embeddings/txt_emb_species.npy \
  --species_names_json data/species_embeddings_scleractinia/embeddings/txt_emb_species.json \
  --batch-size 64 --workers 4 --precision fp32
```
- Notes:
  - Identical results with/without `--hard_restrict_hf` can happen if your genera already aggregate only Scleractinia species (restriction doesn’t change effective prototypes).
  - Coverage and sample counts print before eval; unmatched labels are excluded.

**Closed‑Domain (Your Label Set Only)**

Script: `src/evaluation/closed_domain_eval.py`

- What it does:
  - Builds one text embedding per class from your label strings via templates; softmax over ONLY your label set.
  - Mirrors `zero_shot_iid` template handling with extra options.
- Key flags:
  - `--rank genus|…` & `--rank_only`: choose label form (chain vs token only).
  - `--template_style openai|plain|bio`: prompt set for text encoding.
  - `--normalize_anthozoa`.
  - `--filter_rank / --filter_value`: filter dataset rows (e.g., `order=Scleractinia`).
  - Dumps: `--dump_predictions` (JSON), `--dump_misclassified_csv` (CSV of `path, pred_label, true_label`).
- Typical genus command:
```
python -m src.evaluation.closed_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --csv /path/to/val.csv \
  --rank genus --template_style openai --normalize_anthozoa \
  --batch-size 64 --precision fp32
```
- Observations:
  - Chain‑to‑genus + OpenAI templates > rank‑only on your data.
  - Closed‑domain trails open‑domain (species fan‑out) because there’s only one prototype per genus.

**What Matches the HF Demo Best**

- Open‑domain with species fan‑out + aggregation to your target rank (and hard restriction to corals if desired).
- For display parity, compare top‑k `pred_species_chain` strings to the demo.

**Known Good Settings (on your set)**

- Open‑domain genus, `--rank_only`, `--normalize_anthozoa`, hard‑restricted Scleractinia: ~47/70/82 (top‑1/3/5).
- Closed‑domain genus (chain + OpenAI templates): lower but stable; rank‑only performed worse.

**Troubleshooting & Tips**

- Multiline shell: no trailing spaces after `\` line continuations.
- Ensure dump paths include a directory component; writers create parents as needed.
- 59/60 vs 60/60 coverage: chain mismatches (Anthozoa vs Hexacorallia). Use `--normalize_anthozoa` and/or `--rank_only` for matching.
- Species embeddings: lower batch if OOM; `--hard_restrict_hf` reduces compute/memory for subset scoring.

**Next Steps**

- Closed‑domain “classifier from species”: build class prototypes by averaging HF species text embeddings per genus; evaluate closed‑set with those prototypes (closes the gap to open‑domain).
- Fine‑tuning per `finetune.md`: contrastive at genus with chain prompts, freeze text, unfreeze projector + last K blocks, class‑balanced sampling. Evaluate both closed‑domain and open‑domain hard‑restricted to ensure gains generalize.

