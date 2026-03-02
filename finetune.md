**Goal**

- Fine‑tune BioCLIP‑2 on your coral data (CSV + images) to measurably improve genus‑level accuracy on your existing test set, and maintain parity with demo‑style (open‑domain) evaluation via species fan‑out and aggregation.

**Scope**

- Primary target: Genus classification (report top‑1/3/5). Species optional later (aggregate species→genus at eval).
- Domain slice: Corals (Order = Scleractinia) for first pass. Octocorallia can be considered later.

**Data Assumptions**

- CSV with at least: `Path` (image path), taxonomy columns: `kingdom, phylum, class (or cls), order, family, genus, species`.
- Class naming inconsistencies: use Anthozoa normalization (map Class=Hexacorallia under Cnidaria → Anthozoa) for consistency with TreeOfLife taxonomy.

**1) Prepare Splits**

- Filter (optional for first pass): Order=Scleractinia on both train and val.
- Normalize taxonomy: Anthozoa mapping as above.
- Split: stratified by `genus` with 80/20 split; ensure ≥ k train images per genus (move classes with <3 images to val or skip).
- Save: `train.csv`, `val.csv` (same schema, no extra columns required).

Quick helper to create training CSVs the trainer expects (two columns: `filepath,title`):

```
python - <<'PY'
import pandas as pd, os
src="/path/to/your_coral_full.csv"  # has columns: Path, kingdom, phylum, class, order, family, genus, species
df=pd.read_csv(src)
# Normalize: class=Hexacorallia under Cnidaria -> Anthozoa
if 'class' in df.columns:
    m=(df['phylum'].str.lower()=='cnidaria')&(df['class'].str.lower()=='hexacorallia')
    df.loc[m,'class']='Anthozoa'
# Keep Scleractinia (first pass)
df=df[df['order'].str.lower()=='scleractinia'].copy()
# Drop very-rare genera (<5 images)
g=df['genus'].astype(str).str.strip()
keep=g.value_counts()[lambda s:s>=5].index
df=df[g.isin(keep)].copy()
# Chain-to-genus caption
tax=['kingdom','phylum','class','order','family','genus']
df['title']=df[tax].astype(str).apply(lambda r: " ".join([x for x in r.tolist() if x and x!='nan']), axis=1)
df['filepath']=df['Path'].apply(os.path.abspath)
# Stratified 80/20 by genus
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1337)
y=g.loc[df.index]
train_idx, val_idx = next(sss.split(df, y))
train, val = df.iloc[train_idx], df.iloc[val_idx]
os.makedirs('data', exist_ok=True)
train[['filepath','title']].to_csv('data/coral_train.csv', index=False)
val[['filepath','title']].to_csv('data/coral_val.csv', index=False)
print('Train/Val:', len(train), len(val), 'classes:', train['genus'].nunique())
PY
```

**2) Establish Baselines**

- Open‑domain (demo‑like), hard‑restricted to Scleractinia:
  - Use `src/evaluation/open_domain_eval.py` with `--restrict_hf_rank order --restrict_hf_value Scleractinia --hard_restrict_hf`.
  - Report top‑1/3/5 at genus with `--rank genus --rank_only`.
- Closed‑domain (prompt‑based) on your label set:
  - Use `src/evaluation/closed_domain_eval.py` with `--rank genus --template_style openai --normalize_anthozoa`.
- Linear‑probe (optional quick upper bound):
  - Extract frozen image features, train a logistic regression per genus. Target: confirm headroom and detect under/overfitting risks.

**3) Fine‑Tuning Strategy**

- 3A. Contrastive fine‑tune (image–text) at genus level (recommended start)
  - Text prompt per image: taxonomy chain‑to‑genus (e.g., “Animalia Cnidaria Anthozoa Scleractinia … Genus”).
  - Template ensemble: OpenAI + “bio”; optionally add “coral” domain prompts.
  - Freeze: entire text tower.
  - Unfreeze: visual projector + last 2–4 vision transformer blocks.
  - Optimizer: AdamW; LR 1e‑5 to 5e‑5 for unfrozen params; WD 0.01.
  - Scheduler: cosine with 500 warmup steps.
  - Epochs: 5–10; batch size 64 (tune as per GPU), AMP mixed precision.
  - Augment: use model’s train preprocess; avoid heavy color shifts (underwater imagery sensitivity).
  - Imbalance: class‑balanced sampler or per‑class loss weights.

- 3B. Projector‑only fine‑tune (fallback if overfitting)
  - Freeze all but visual projector.
  - Same optimizer/schedule, fewer epochs (3–5).

- 3C. Species‑level contrastive (optional second phase)
  - Train with species binomials (scientific names) as text; at eval, aggregate species→genus like the demo.

**4) Implementation in This Repo**

- Add `src/finetune/coral_contrastive.py` (to be implemented):
  - Load model via `create_model_and_transforms("hf-hub:imageomics/bioclip-2")`.
  - Build train/val datasets from CSVs via `FranCatalogueLoader` with `rank="genus"`, `rank_only=False`, `normalize_anthozoa=True`.
  - Build N text prompts per image (OpenAI + bio [+ coral]) from the chain‑to‑genus; encode and average text embeddings (normalized).
  - Encode image; compute CLIP InfoNCE loss over batch.
  - Freeze/unfreeze as per plan; optimizer, scheduler, AMP.
  - Checkpoint best by val top‑1 genus; log curves.

**5) Filtering & Sampling**

- Optionally filter dataset to Scleractinia at dataset build time (consistent with eval).
- Use class‑balanced sampler for train to mitigate skewed genus distribution.

**6) Evaluation After Fine‑Tune**

- Closed‑domain (prompt‑based):
  - Evaluate on `val.csv` with `src/evaluation/closed_domain_eval.py` (
    `--rank genus --template_style openai --normalize_anthozoa`).
  - Optionally add a flag to load your checkpoint; or evaluate inside `coral_contrastive.py` after each epoch.
- Open‑domain (demo‑like), hard‑restricted:
  - Reuse `src/evaluation/open_domain_eval.py` with `--hard_restrict_hf` and Scleractinia restriction.
  - Optionally extend the script to load a local checkpoint, or evaluate inside `coral_contrastive.py` using the same code path.

**7) Guardrails / Early Stop**

- Tune LR conservatively; early stop on val top‑1.
- Keep a small held‑out test set unseen during fine‑tune for final report.
- Use light augmentations only; avoid distorting morphology and color.

**8) Prompt Engineering (Iterative)**

- Start with chain‑to‑genus + OpenAI templates.
- Try adding `bio` templates (already supported): "{}", "a photo of {}", "an image of {}", "a biological photograph of {}".
- Add a `coral` domain set (to implement): e.g., "a stony coral: {}", "a scleractinian coral: {}", "an underwater photo of coral {}".
- Prompt‑mix: average embeddings from chain‑to‑genus and rank‑only genus forms per class.

**9) Practical Command Sketches**

- Finetune (to be implemented):
  - `python -m src.finetune.coral_contrastive \
    --model hf-hub:imageomics/bioclip-2 \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --rank genus \
    --normalize_anthozoa \
    --template_style openai bio coral \
    --freeze_text \
    --unfreeze_last_k 3 \
    --lr 5e-5 --wd 0.01 --epochs 10 --batch-size 64 --accum_steps 2 \
    --precision amp \
    --save_dir ./checkpoints/coral_genus`

- Evaluate closed‑domain (prompt‑based):
  - `python -m src.evaluation.closed_domain_eval \
    --model hf-hub:imageomics/bioclip-2 \
    --csv /path/to/val.csv \
    --rank genus \
    --template_style openai \
    --normalize_anthozoa \
    --batch-size 64 --precision fp32 \
    --dump_misclassified_csv ./logs/closed_domain/val_misclassified.csv`

- Evaluate open‑domain, hard‑restricted Scleractinia:
  - `python -m src.evaluation.open_domain_eval \
    --model hf-hub:imageomics/bioclip-2 \
    --csv /path/to/val.csv \
    --rank genus \
    --rank_only \
    --normalize_anthozoa \
    --restrict_hf_rank order \
    --restrict_hf_value Scleractinia \
    --hard_restrict_hf \
    --species_emb_npy data/species_embeddings_scleractinia/embeddings/txt_emb_species.npy \
    --species_names_json data/species_embeddings_scleractinia/embeddings/txt_emb_species.json \
    --batch-size 64 --workers 4 --precision fp32`

**10) Milestones**

- M1: Fine‑tune loop runs on a subset; val top‑1 improves by ≥2–5 pts (closed‑domain).
- M2: Open‑domain hard‑restricted improves by ≥2–5 pts on the test CSV.
- M3: Lock best checkpoint; document hyperparams.
- M4 (optional): Species‑level fine‑tune; evaluate genus via aggregation.

**11) Risks & Mitigations**

- Overfitting small classes → use balanced sampling, low LR, early stop.
- Prompt mismatch → iterate `template_style` and add coral prompts; use prompt‑mix.
- Taxonomy mismatch → keep Anthozoa normalization; rank‑only matching where appropriate.
- Domain gap to demo → validate both closed‑domain and open‑domain (hard‑restricted) to ensure gains generalize.

**12) Next Engineering Tasks**

- [ ] Implement `src/finetune/coral_contrastive.py` with the above loop.
- [ ] Add `--load_checkpoint` support to `open_domain_eval.py` and/or `closed_domain_eval.py` to evaluate finetuned weights.
- [ ] Add `template_style coral` and prompt‑mix option to `closed_domain_eval.py`.
- [ ] (Optional) Linear probe helper for a frozen‑encoder upper bound.

**Appendix: Using the Existing Trainer (no new code)**

- The trainer in `src/training/main.py` supports CSV datasets out of the box. Once `data/coral_train.csv` and `data/coral_val.csv` (columns: `filepath,title`) are ready, fine‑tune with:

```
python -m src.training.main \
  --model hf-hub:imageomics/bioclip-2 \
  --dataset-type csv \
  --csv-separator , \
  --csv-img-key filepath \
  --csv-caption-key title \
  --train-data data/coral_train.csv \
  --val-data data/coral_val.csv \
  --batch-size 64 --workers 8 \
  --epochs 10 \
  --lr 5e-5 --wd 0.01 \
  --precision amp \
  --logs ./logs_finetune/coral_genus \
  --name coral_genus_ft \
  --save-frequency 1 --val-frequency 1
```

- Optional freezing (safer first pass): add `--lock-image --lock-image-unlocked-groups 2` to keep most of the image tower frozen, unfreezing only the last 2 groups + projector.
- If OOM: reduce `--batch-size` (e.g., 32/16) or use fewer workers.

**Targets**

- Short‑term: +5–15 top‑1 (genus) on open‑domain hard‑restricted (from ~47 → 60+). Closed‑domain should follow.
- Ambitious: 80–90 top‑1 is achievable with species‑level fine‑tune + aggregation and/or more data; validate on a held‑out test split.
