# Next Steps: Coral Fine-Tuning Roadmap

## 1. Benchmark Current Model
- Run `src.evaluation.closed_domain_eval` on the latest checkpoint for both genus and species ranks to quantify closed-domain gains.
- Re-evaluate earlier epochs (e.g., 20, 40, 60) on open-domain genus/species to detect possible overfitting.

## 2. Align Prompt Embeddings
- Regenerate text embeddings:
  - Chain-only, rank-only, chain+rank ensemble, and coral-prefixed variants.
  - Sum or average the embeddings and re-run open-domain eval to compare top-k.
- Note: ensure evaluation `--rank`/`--rank_only` flags match the embedding label format.

## 3. Genus-Focused Fine-Tune
- Launch a new run targeting genus labels:
  - `--coral-target-level genus`
  - `--coral-rank-only`
  - `--coral-caption-mode rank_only`
- Start with text tower frozen; capture checkpoints after 10, 20 epochs for comparison.

## 4. Unfreeze Text Components
- Projector-only fine-tune: unfreeze CLIP text projection and matching vision layers with LR ~5e-6.
- If genus gains remain small, unfreeze top 2–3 text transformer layers (very small LR, e.g., 1e-6).
- Monitor both closed and open-domain metrics after unfreezing.

## 5. Sampling & Augmentation Tweaks
- Confirm balanced sampling per species (already on); optionally test genus-based sampling.
- Add prompt mix during training (chain + rank-only) to prepare text tower for multiple evaluation forms.

## 6. Evaluation Matrix
- Maintain both HF and custom embedding banks; report top-1/3/5 for each.
- Document results for: (a) HF genus, (b) custom genus, (c) HF species, (d) custom species, (e) closed-domain genus & species.

## 7. Documentation & Tracking
- Track configurations (run name, checkpoint, LR schedule tweaks, prompt style) for each experiment.
- Record improvement deltas over baseline; identify best run for production embedding regeneration once genus top-1 ≥ 65%.

---

Primary target: increase open-domain genus accuracy from ~56 → 65+. Secondary: push species open-domain beyond 40% while retaining strong closed-domain results.
