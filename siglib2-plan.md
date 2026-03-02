# SigLIP-2 Coral Adaptation Plan

## Goals
- Replace BioCLIP/BioCLIP-2 backbone with SigLIP-2 to leverage its stronger multilingual vision-language training and dense features.
- Fine-tune SigLIP-2 on coral taxonomy captions to maximize genus/species accuracy.
- Evaluate across all existing datasets (Fran catalogue, RSG, Acropora) in both closed-domain and open-domain settings.

## Resources
- Paper: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features" (arXiv:2502.14786)
- Checkpoints: https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text/README_siglip2.md
- Training pipeline: reuse `src/training/main.py` (contrastive image-text), `coral_split` dataset loader, existing evaluation scripts.

## Plan Overview
1. Prepare SigLIP-2 model configs and integrate into repository.
2. Convert and load SigLIP-2 weights for use in our training/eval scripts.
3. Align tokenizer/multilingual text tower with coral captions.
4. Stage fine-tuning experiments (species, genus, chain vs rank-only, projector/text unfreeze).
5. Evaluate across closed/open domain, compare to BioCLIP-2 baselines, and iterate on improvements (embeddings, prompt mixes, localization extraction).

---

## 1. Model Integration
- [ ] Download SigLIP-2 checkpoints for desired sizes (start with ViT-L/14; consider ViT-B/32 for quick tests).
- [ ] Port SigLIP-2 model config (vision tower architecture, text tower, MAP head) into `src/open_clip/model_configs` or create a new loader.
- [ ] Confirm compatibility with `create_model_and_transforms` (ensure `pretrained` string resolves to local checkpoint; no `load_weights_only`).
- [ ] Update tokenizer: SigLIP-2 uses a multilingual Gemma tokenizer with 256k vocab – ensure we import or convert this tokenizer for `get_tokenizer`.
- [ ] Confirm image preprocessing matches SigLIP-2 (resolution, aspect ratio handling). Start with fixed 224×224; later explore NaFlex variant if needed.

## 2. Coral Caption Alignment
- [ ] Build CSVs with taxonomy captions at genus and species levels (chain and rank-only). Ensure multilingual readiness (currently coral names are Latin; no issues).
- [ ] Run initial zero-shot with SigLIP-2 before fine-tuning to establish baseline on coral val/test sets.
- [ ] Implement prompt ensembles (chain, rank-only, coral-specific prefixes) – SigLIP-2 can handle longer prompts due to stronger text tower.

## 3. Fine-Tuning Stages
**Stage A – Species Chain Baseline**
- [ ] Fine-tune SigLIP-2 with `--coral-target-level species`, chain captions, text tower frozen.
- [ ] Evaluate closed domain & open domain (HF embeddings + custom species/genus banks). Record baseline.

**Stage B – Genus Rank-Only**
- [ ] Run genus rank-only fine-tune to match open-domain genus target. Compare to Stage A.

**Stage C – Projector/Text Unfreeze**
- [ ] Enable `--unlock-text-projection` with low LR (e.g., 5e-6) to align text embeddings with coral taxonomy.
- [ ] Optionally unfreeze last N text layers (very small LR) if projector-only update is insufficient.

**Stage D – Species + Genus Mix**
- [ ] Mix species and genus captions during training (augment dataset with both forms) to let SigLIP-2 learn shared prompts.
- [ ] Evaluate whether combined training improves aggregated genus performance.

## 4. Localization & Dense Features
- [ ] Explore SigLIP-2’s dense feature capability: extract attention maps or dense embeddings for coral images, evaluate localization (e.g., highlighting colony structures).
- [ ] Consider lightweight decoder fine-tune on a subset of coral images with bounding boxes, leveraging SigLIP-2’s native support for localization tasks.

## 5. Evaluation & Embeddings
- [ ] Use `tools/export_coral_embeddings.py` to regenerate embedding banks from SigLIP-2 checkpoints after each fine-tune.
- [ ] Run `tools/eval_checkpoints.py` to batch evaluate closed/open domain metrics across checkpoints.
- [ ] Compare against BioCLIP-2 baselines; target: genus open-domain ≥65%, species ≥40%.
- [ ] Verification on RSG and Acropora datasets to ensure transfer gains.

## 6. Iteration & Documentation
- [ ] Document hyperparameters, best checkpoints, and embedding files for SigLIP-2 runs.
- [ ] Track improvement deltas (zero-shot vs fine-tune, species vs genus) in a summary table.
- [ ] If SigLIP-2 underperforms, investigate additional training signals (self-distillation, masked prediction) by mimicking parts of the SigLIP-2 recipe.

## Stretch Goals
- [ ] Integrate localization decoder (e.g., LocCa) to directly leverage SigLIP-2’s captioning-based training for coral part segmentation.
- [ ] Build an explainability tool (Finer-CAM) on top of SigLIP-2 features to visualize discriminative coral regions.
- [ ] Explore multilingual captions (e.g., coral common names in different languages) to take advantage of SigLIP-2’s multilingual text tower.

---

### Milestones
1. **M0:** SigLIP-2 zero-shot baseline recorded on coral datasets.
2. **M1:** Species-chain fine-tune surpasses BioCLIP-2 species baseline.
3. **M2:** Genus rank-only fine-tune achieves ≥65% open-domain genus accuracy.
4. **M3:** Best checkpoint & embedding bank documented; evaluation scripts updated.
5. **M4 (stretch):** Localization/explainability demo using SigLIP-2 whether dense features emphasize coral anatomy.

