"""
Open-domain, demo-like evaluation over a closed label set (CSV or folder).

Pipeline:
- Load BioCLIP-2 model and preprocess
- Load species text embeddings + names (from HF dataset, locally cached files)
- Load samples via FranCatalogueLoader (--dataset_loader=csv), AcroporaDataset (--dataset_loader=acropora), or RSGPatchDataset (--dataset_loader=rsg) at a chosen rank (default: genus)
- Match dataset labels to aggregated species labels (rank-only for robust matching)
- Drop samples whose labels are not covered (e.g., taxonomy mismatch -> 59/60)
- For each batch: encode image -> score vs all species -> softmax -> aggregate to labels
- Compute top-1/3/5 accuracy on the filtered set

Example (genus, rank-only, normalized Anthozoa):

python -m src.evaluation.open_domain_eval \
  --model hf-hub:imageomics/bioclip-2 \
  --dataset_loader csv \
  --csv /home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv \
  --rank genus \
  --species_emb_npy bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.npy \
  --species_names_json bioclip2_demo/bioclip-2-demo/components/data/species_embeddings/embeddings/txt_emb_species.json \
  --batch-size 64 --workers 4

"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .zero_shot_iid import accuracy
from ..open_clip import create_model_and_transforms, get_cast_dtype
from ..training.precision import get_autocast
from dataset_catalogue import FranCatalogueLoader
from dataset_acropora import AcroporaDataset
from dataset_rsg import RSGPatchDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--dataset_loader", type=str, default="csv", choices=["csv", "acropora", "rsg"], help="Which dataset loader to use: csv (FranCatalogue), acropora (folder-based), rsg (patch CSV)")
    ap.add_argument("--csv", type=str, help="Path to your dataset CSV (FranCatalogue or RSG)")
    ap.add_argument("--data_root", type=str, default=None, help="Base dir for relative image paths (CSV loader only)")
    ap.add_argument("--acropora_root", type=str, default=None, help="Root directory for AcroporaDataset when --dataset_loader=acropora")
    ap.add_argument("--acropora_split", type=str, default="train", choices=["train", "val"], help="Split to use for AcroporaDataset defaults")
    ap.add_argument("--rsg_taxonomy_csv", type=str, default=None, help="Taxonomy CSV for RSG loader (defaults to dataset setting)")
    ap.add_argument("--rsg_corals_only", action="store_true", help="Filter RSG CSV to coral genera subset")
    ap.add_argument("--rsg_path_key", type=str, default=None, help="Override image-path column in RSG CSV")
    ap.add_argument("--rsg_label_key", type=str, default=None, help="Override label column in RSG CSV")
    ap.add_argument("--rank", type=str, default="genus", choices=["kingdom","phylum","cls","order","family","genus","species"])
    ap.add_argument("--rank_only", action="store_true", help="Use only the selected rank as label text (recommended for matching)")
    ap.add_argument("--normalize_anthozoa", action="store_true", help="Map class=Hexacorallia under Cnidaria to Anthozoa for matching")
    ap.add_argument("--require_species", action="store_true", help="Drop samples missing species epithet when evaluating at species rank")
    ap.add_argument("--species_emb_npy", type=str, required=True)
    ap.add_argument("--species_names_json", type=str, required=True)
    ap.add_argument("--class_list", type=str, default=None, help="Optional file of taxonomy chains to restrict label set")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32","amp","bf16","amp_bfloat16","amp_bf16"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional local checkpoint (.pt) to load before evaluation")
    ap.add_argument("--dump_predictions", type=str, default=None, help="Optional JSON path to dump per-image predictions")
    ap.add_argument("--dump_topk", type=int, default=5)
    ap.add_argument("--dump_misclassified_csv", type=str, default=None, help="If set, write a CSV of misclassified examples with image path and predicted taxonomy string.")
    ap.add_argument("--filter_rank", type=str, default=None, choices=["kingdom","phylum","cls","class","order","family","genus","species"], help="Optional taxonomy rank column to filter the data by.")
    ap.add_argument("--filter_value", type=str, default=None, help="Value to filter at the given --filter_rank.")
    ap.add_argument("--restrict_hf_rank", type=str, default=None, choices=["kingdom","phylum","cls","class","order","family","genus","species"], help="Restrict the HF species pool to a specific rank value before aggregation.")
    ap.add_argument("--restrict_hf_value", type=str, default=None, help="Value to restrict at the given --restrict_hf_rank.")
    ap.add_argument("--hard_restrict_hf", action="store_true", help="Physically slice species embeddings and names to the restricted pool before scoring.")
    return ap.parse_args()


def load_species_text(species_emb_npy: str, species_names_json: str, device: torch.device, cast_dtype=None) -> Tuple[torch.Tensor, list]:
    emb = np.load(species_emb_npy)
    species_emb = torch.from_numpy(emb).to(device)
    if cast_dtype is not None:
        species_emb = species_emb.to(dtype=cast_dtype)
    with open(species_names_json, 'r') as f:
        names = json.load(f)
    return species_emb, names



def label_from_taxonomy(tax: List[str], rank: str, rank_only: bool) -> str:
    level_map = {"kingdom": 0, "phylum": 1, "cls": 2, "class": 2, "order": 3, "family": 4, "genus": 5, "species": 6}
    rank = rank.lower()
    idx = level_map.get(rank, 5)
    padded = (tax + [""] * 7)[:7]
    if rank_only:
        if rank == "species":
            genus = padded[5].strip()
            species = padded[6].strip()
            return f"{genus} {species}".strip()
        return padded[idx].strip()
    else:
        if rank == "species":
            genus = padded[5].strip()
            species = padded[6].strip()
            parts = [p.strip() for p in padded[:5] if p.strip()]
            if genus or species:
                parts.append(f"{genus} {species}".strip())
            return " ".join(parts).strip()
        parts = [p.strip() for p in padded[: idx + 1] if p.strip()]
        return " ".join(parts).strip()


def load_class_list(class_file: str, rank: str, rank_only: bool) -> List[str]:
    labels = []
    with open(class_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            label = label_from_taxonomy(parts, rank, rank_only)
            if label:
                labels.append(label)
    return labels


def apply_class_list(ds, desired_labels: List[str]):
    desired_set = {lbl: i for i, lbl in enumerate(desired_labels)}
    new_paths, new_targets = [], []
    seen = set()
    for path, idx in ds.samples:
        label = ds.idx_to_class[idx]
        if label in desired_set:
            new_paths.append(path)
            new_targets.append(desired_set[label])
            seen.add(label)
    if not new_paths:
        print("[WARN] No samples matched the provided class list.")
    missing = [lbl for lbl in desired_labels if lbl not in seen]
    if missing:
        print(f"[WARN] class_list labels without samples: {missing}")
    ds.paths = new_paths
    ds.targets = new_targets
    ds.samples = list(zip(ds.paths, ds.targets))
    ds.classes = desired_labels
    ds.class_to_idx = desired_set
    ds.idx_to_class = {i: lbl for lbl, i in desired_set.items()}
    ds.labels = [ds.idx_to_class[t] for t in ds.targets]
    if hasattr(ds, 'data'):
        cols = {c.lower(): c for c in ds.data.columns}
        path_col = cols.get('path')
        if path_col:
            path_set = {str(p) for p in ds.paths}
            ds.data = ds.data[ds.data[path_col].astype(str).isin(path_set)].reset_index(drop=True)
    return ds

def build_mapping(names: list, rank: str, allowed_labels: List[str], rank_only: bool) -> Tuple[Dict[str, List[int]], List[str]]:
    level_map = {"kingdom":0,"phylum":1,"cls":2,"class":2,"order":3,"family":4,"genus":5,"species":6}
    idx = level_map[rank]

    def aggregated_label(entry):
        tax = entry[0]
        if rank_only:
            if idx == 6 and len(tax) >= 7:
                return f"{tax[5]} {tax[6]}".strip()
            return tax[idx]
        else:
            if idx == 6 and len(tax) >= 7:
                parts = [p.strip() for p in tax[:6] if isinstance(p, str) and p.strip()]
                species = tax[6].strip() if isinstance(tax[6], str) else str(tax[6]).strip()
                if species:
                    parts.append(species)
                return " ".join(parts)
            return " ".join(p.strip() for p in tax[:idx+1] if isinstance(p, str) and p.strip())

    label_per_species = [aggregated_label(e) for e in names]
    mapping = {}
    for i, lab in enumerate(label_per_species):
        mapping.setdefault(lab, []).append(i)

    allowed = set(allowed_labels)
    mapping = {lab: idxs for lab, idxs in mapping.items() if lab in allowed}
    label_order = [lab for lab in allowed_labels if lab in mapping]
    return mapping, label_order


def filter_and_remap_dataset(ds, label_order: List[str]):
    # Build new class_to_idx with label_order
    new_c2i = {c: i for i, c in enumerate(label_order)}
    new_targets = []
    new_paths = []
    new_labels = []
    for path, old_idx in ds.samples:
        cls_name = ds.idx_to_class[old_idx]
        if cls_name in new_c2i:
            new_paths.append(path)
            new_targets.append(new_c2i[cls_name])
            new_labels.append(cls_name)

    # Mutate ds in-place (safe for this script)
    ds.class_to_idx = new_c2i
    ds.idx_to_class = {i: c for c, i in new_c2i.items()}
    ds.classes = label_order
    ds.paths = new_paths
    ds.targets = new_targets
    ds.labels = new_labels
    ds.samples = list(zip(ds.paths, ds.targets))
    if hasattr(ds, 'data'):
        cols = {c.lower(): c for c in ds.data.columns}
        path_col = cols.get('path')
        if path_col:
            mask = ds.data[path_col].astype(str).isin({str(p) for p in new_paths})
            ds.data = ds.data[mask].reset_index(drop=True)
    return ds


def main():
    args = parse_args()

    device = torch.device(args.device)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=None,
        precision=args.precision,
        device=device,
        output_dict=True,
        load_weights_only=True,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt

        if isinstance(state, dict) and state and any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")
        print(f"Loaded checkpoint: {args.checkpoint}")

    cast_dtype = get_cast_dtype(args.precision)
    species_emb, names = load_species_text(args.species_emb_npy, args.species_names_json, device=device, cast_dtype=cast_dtype)

    # Optional hard restriction: slice species_emb and names to a taxonomy subset
    if args.restrict_hf_rank and args.restrict_hf_value and getattr(args, 'hard_restrict_hf', False):
        level_map = {"kingdom":0,"phylum":1,"cls":2,"class":2,"order":3,"family":4,"genus":5,"species":6}
        rkey = args.restrict_hf_rank.lower()
        r_idx = level_map[rkey]
        target_val = str(args.restrict_hf_value).lower()
        keep = [i for i,e in enumerate(names) if str(e[0][r_idx]).lower() == target_val]
        if not keep:
            raise RuntimeError(f"No species matched {args.restrict_hf_rank} == {args.restrict_hf_value}")
        species_emb = species_emb[:, keep]
        names = [names[i] for i in keep]
        print(f"Hard-restricted HF species: kept {len(keep)} entries; embeddings shape {tuple(species_emb.shape)}")

    # Load dataset at requested rank; supports CSV (Fran catalogue), folder (Acropora), or RSG patches
    taxonomy_filters = None
    if args.filter_rank and args.filter_value and args.dataset_loader == "csv":
        rank_key = args.filter_rank.lower()
        if rank_key == 'class':
            rank_key = 'cls'
        taxonomy_filters = {rank_key: args.filter_value}

    rank_only_flag = args.rank_only
    if args.dataset_loader == "csv":
        if not args.csv:
            raise ValueError("--csv is required when --dataset_loader=csv")
        ds = FranCatalogueLoader(
            csv_file=args.csv,
            data_root=args.data_root,
            taxonomic_level=args.rank,
            transform=preprocess_val,
            require_species=args.require_species,
            rank_only=rank_only_flag,
            normalize_anthozoa=args.normalize_anthozoa,
            taxonomy_filters=taxonomy_filters,
        )
    elif args.dataset_loader == "acropora":
        ds = AcroporaDataset(
            data_root=args.acropora_root,
            split=args.acropora_split,
            taxonomic_level=args.rank,
            transform=preprocess_val,
            require_species=args.require_species,
            rank_only=rank_only_flag,
            normalize_anthozoa=args.normalize_anthozoa,
        )
    else:
        if not args.csv:
            raise ValueError("--csv is required when --dataset_loader=rsg")
        rsg_kwargs = {}
        if args.rsg_taxonomy_csv:
            rsg_kwargs['taxonomy_csv'] = args.rsg_taxonomy_csv
        if args.rsg_path_key:
            rsg_kwargs['path_key'] = args.rsg_path_key
        if args.rsg_label_key:
            rsg_kwargs['label_key'] = args.rsg_label_key
        ds = RSGPatchDataset(
            csv_file=args.csv,
            taxonomic_level=args.rank,
            transform=preprocess_val,
            require_species=args.require_species,
            rank_only=rank_only_flag,
            normalize_anthozoa=args.normalize_anthozoa,
            corals_only=args.rsg_corals_only,
            **rsg_kwargs,
        )
    # Optional dataset filtering by a taxonomy rank (e.g., only Order=Scleractinia)
    filter_applied_in_loader = args.dataset_loader == 'csv' and taxonomy_filters is not None
    if args.filter_rank and args.filter_value and not filter_applied_in_loader:
        rank_key = args.filter_rank.lower()
        if rank_key == 'class':
            rank_key = 'cls'
        if hasattr(ds, 'data') and rank_key in {c.lower() for c in ds.data.columns}:
            cols = {c.lower(): c for c in ds.data.columns}
            col = cols[rank_key]
            mask = ds.data[col].astype(str).str.lower() == str(args.filter_value).lower()
            keep_idx = [i for i, m in enumerate(mask.tolist()) if m]
            ds.paths = [ds.paths[i] for i in keep_idx]
            ds.targets = [ds.targets[i] for i in keep_idx]
            ds.samples = list(zip(ds.paths, ds.targets))
            ds.labels = [ds.idx_to_class[t] for t in ds.targets]
            present = sorted({ds.idx_to_class[t] for t in ds.targets})
            ds.classes = present
            if hasattr(ds, 'data'):
                cols = {c.lower(): c for c in ds.data.columns}
                path_col = cols.get('path')
                if path_col:
                    ds.data = ds.data.iloc[keep_idx].reset_index(drop=True)
            print(f"Applied filter: {args.filter_rank} == {args.filter_value}. Remaining samples: {len(ds.paths)}")
        else:
            print(f"[WARN] filter_rank '{args.filter_rank}' not found in dataset columns; skipping filter.")

    # Optional class list restriction (taxonomy chains -> desired labels)
    if args.class_list:
        desired_labels = load_class_list(args.class_list, args.rank, rank_only_flag)
        if not desired_labels:
            raise ValueError(f"No labels found in class_list file: {args.class_list}")
        ds = apply_class_list(ds, desired_labels)
        print(f"Applied class list restriction: {len(desired_labels)} labels")

    # Build mapping from species -> aggregated label; restrict to dataset classes
    mapping, label_order = build_mapping(names, args.rank, ds.classes, rank_only=rank_only_flag)

    # Optionally restrict HF open-domain labelset by taxonomy (e.g., only Order=Scleractinia)
    if args.restrict_hf_rank and args.restrict_hf_value:
        level_map = {"kingdom":0,"phylum":1,"cls":2,"class":2,"order":3,"family":4,"genus":5,"species":6}
        rkey = args.restrict_hf_rank.lower()
        r_idx = level_map[rkey]
        target_val = str(args.restrict_hf_value).lower()
        kept = []
        for lab in list(label_order):
            sp_idxs = mapping.get(lab, [])
            keep_lab = any(str(names[i][0][r_idx]).lower() == target_val for i in sp_idxs)
            if keep_lab:
                kept.append(lab)
        mapping = {lab: mapping[lab] for lab in kept}
        label_order = kept
        print(f"Restricted HF labelset: kept {len(label_order)} labels matching {args.restrict_hf_rank} == {args.restrict_hf_value}")

    # Report coverage
    orig_samples = len(ds)
    orig_classes = len(ds.classes)
    covered = set(mapping.keys())
    missing = [c for c in ds.classes if c not in covered]
    print(f"Coverage: {len(ds.classes)-len(missing)}/{len(ds.classes)} labels matched")
    if missing:
        print("Missing (first 10):", missing[:10])

    # Drop samples for missing labels and remap targets to new indices
    ds = filter_and_remap_dataset(ds, label_order)
    print(f"Evaluating on {len(ds)} samples across {len(ds.classes)} classes (from {orig_samples} samples, {orig_classes} classes)")

    # Dataloader
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)

    # Aggregation matrix A: species -> labels
    A = torch.zeros((species_emb.shape[1], len(label_order)), dtype=torch.float32, device=device)
    for j, lab in enumerate(label_order):
        idxs = mapping[lab]
        A[idxs, j] = 1.0

    autocast = get_autocast(args.precision)
    dump = args.dump_predictions is not None
    dump_rows = []
    for_topk = (1, min(3, len(label_order)), min(5, len(label_order)))
    topk_acc = {k: 0.0 for k in for_topk}
    n = 0
    # prepare misclassified CSV collection
    collect_mis = args.dump_misclassified_csv is not None
    mis_rows = []
    # track global index for paths
    global_index = 0
    topk_acc = {k: 0.0 for k in for_topk}
    n = 0

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            targets = targets.to(device)
            with autocast():
                feats, _ = model.encode_image(images)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                logits_species = model.logit_scale.exp() * feats @ species_emb
                probs_species = torch.nn.functional.softmax(logits_species, dim=-1)
                probs_rank = probs_species @ A

            # compute accuracy using probabilities as scores
            acc = accuracy(probs_rank, targets, topk=for_topk)
            for k, v in acc.items():
                topk_acc[k] += v
            n += images.size(0)

            if dump or collect_mis:
                bs = images.size(0)
                for bi in range(bs):
                    if dump:
                        pvals, pidx = torch.topk(probs_rank[bi], k=min(args.dump_topk, probs_rank.shape[-1]))
                        preds = [
                            {"rank": j+1, "index": int(pidx[j].item()), "label": label_order[int(pidx[j].item())], "prob": float(pvals[j].item())}
                            for j in range(pidx.numel())
                        ]
                        dump_rows.append({
                            "true_index": int(targets[bi].item()),
                            "true_label": label_order[int(targets[bi].item())],
                            "topk": preds,
                        })
                    if collect_mis:
                        true_idx = int(targets[bi].item())
                        pred_idx = int(torch.argmax(probs_rank[bi]).item())
                        if pred_idx != true_idx:
                            path = ds.paths[global_index + bi]
                            pred_label = label_order[pred_idx]
                            sp_indices = mapping[pred_label]
                            sp_scores = probs_species[bi][sp_indices]
                            best_local = int(torch.argmax(sp_scores).item())
                            sp_idx = sp_indices[best_local]
                            taxon, common = names[sp_idx]
                            # Build display strings: genus-level chain and full species chain
                            genus_chain = " ".join(taxon[:6])
                            species_display = " ".join(taxon)
                            if common:
                                species_display = f"{species_display} ({common})"
                            mis_rows.append((path, genus_chain, species_display))
                global_index += bs

    if n == 0:
        print("No samples matched the evaluation criteria (coverage=0). Skipping accuracy computation.")
        return

    for k in list(topk_acc.keys()):
        topk_acc[k] /= float(n)

    print("Results:")
    for k in sorted(topk_acc.keys()):
        print(f"  top{k}: {topk_acc[k]*100:.2f}")

    if dump:
        os.makedirs(os.path.dirname(args.dump_predictions), exist_ok=True)
        with open(args.dump_predictions, 'w') as f:
            json.dump(dump_rows, f, indent=2)
        print("Wrote:", args.dump_predictions)
    if collect_mis:
        import csv
        dirn = os.path.dirname(args.dump_misclassified_csv)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        with open(args.dump_misclassified_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["path", "pred_genus_chain", "pred_species_chain"])
            for row in mis_rows:
                w.writerow(row)
        print("Wrote:", args.dump_misclassified_csv)


if __name__ == "__main__":
    main()
