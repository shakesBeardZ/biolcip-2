from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_catalogue import FranCatalogueLoader
from dataset_acropora import AcroporaDataset
from dataset_rsg import RSGPatchDataset
from src.open_clip import create_model_and_transforms, get_cast_dtype, get_tokenizer
from src.training.precision import get_autocast
from src.training.imagenet_zeroshot_data import openai_imagenet_template
from src.evaluation.zero_shot_iid import accuracy

DATASET_CHOICES = ["catalogue", "acropora", "rsg"]


def natural_key(string_: str):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch evaluate checkpoints (open/closed domain).")
    ap.add_argument("--checkpoints-dir", required=True, help="Directory containing checkpoint files (epoch_X.pt)")
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "amp", "bf16", "amp_bfloat16", "amp_bf16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-checkpoints", type=int, default=None, help="Optional limit of checkpoints to evaluate")
    ap.add_argument("--output-csv", default="checkpoint_eval.csv")
    ap.add_argument("--eval-open", action="store_true", help="Run open-domain evaluation")
    ap.add_argument("--eval-closed", action="store_true", help="Run closed-domain evaluation")

    # dataset options shared
    ap.add_argument("--dataset_loader", default="catalogue", choices=DATASET_CHOICES)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--rank", default="genus", choices=["kingdom","phylum","cls","order","family","genus","species","scientific"])
    ap.add_argument("--rank_only", action="store_true")
    ap.add_argument("--normalize_anthozoa", action="store_true")
    ap.add_argument("--filter_rank", default=None, choices=["kingdom","phylum","cls","class","order","family","genus","species"])
    ap.add_argument("--filter_value", default=None)

    # open-domain specific
    ap.add_argument("--species_emb_npy", default=None)
    ap.add_argument("--species_names_json", default=None)
    ap.add_argument("--restrict_hf_rank", default=None, choices=["kingdom","phylum","cls","class","order","family","genus","species"])
    ap.add_argument("--restrict_hf_value", default=None)
    ap.add_argument("--hard_restrict_hf", action="store_true")

    # closed-domain specific
    ap.add_argument("--template_style", default="openai", choices=["openai","plain","bio"])

    # rsg specific
    ap.add_argument("--rsg_taxonomy_csv", default=None)
    ap.add_argument("--rsg_corals_only", action="store_true")
    ap.add_argument("--rsg_path_key", default=None)
    ap.add_argument("--rsg_label_key", default=None)

    args = ap.parse_args()
    if not args.eval_open and not args.eval_closed:
        raise ValueError("Specify at least one of --eval-open or --eval-closed")
    if args.eval_open:
        if not args.species_emb_npy or not args.species_names_json:
            raise ValueError("Open-domain eval requires --species_emb_npy and --species_names_json")
    return args


def list_checkpoints(directory: str, limit: int | None = None) -> List[Path]:
    paths = sorted(Path(directory).glob('*.pt'), key=lambda p: natural_key(p.name))
    if limit:
        paths = paths[:limit]
    return paths


def load_model(args: argparse.Namespace):
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=None,
        precision=args.precision,
        device=args.device,
        output_dict=True,
        load_weights_only=True,
    )
    return model, preprocess_val


def load_checkpoint(model, checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


def build_templates(style: str):
    if style == "plain":
        return [lambda c: f"{c}"]
    if style == "bio":
        return [
            lambda c: f"{c}",
            lambda c: f"a photo of {c}",
            lambda c: f"an image of {c}",
            lambda c: f"a biological photograph of {c}",
        ]
    return openai_imagenet_template


def load_dataset(args: argparse.Namespace, preprocess, for_closed: bool):
    loader = args.dataset_loader
    if loader == "rsg":
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
            transform=preprocess,
            require_species=False,
            rank_only=args.rank_only,
            normalize_anthozoa=args.normalize_anthozoa,
            corals_only=args.rsg_corals_only,
            **rsg_kwargs,
        )
    elif loader == "acropora":
        ds = AcroporaDataset(
            data_root=args.data_root,
            split="val",
            taxonomic_level=args.rank,
            transform=preprocess,
            require_species=False,
            rank_only=args.rank_only,
            normalize_anthozoa=args.normalize_anthozoa,
        )
    else:
        ds = FranCatalogueLoader(
            csv_file=args.csv,
            data_root=args.data_root,
            taxonomic_level=args.rank,
            transform=preprocess,
            rank_only=args.rank_only,
            normalize_anthozoa=args.normalize_anthozoa,
        )

    if args.filter_rank and args.filter_value and hasattr(ds, "data"):
        rank_key = args.filter_rank.lower()
        if rank_key == "class":
            rank_key = "cls"
        cols = {c.lower(): c for c in ds.data.columns}
        if rank_key in cols:
            col = cols[rank_key]
            target_val = str(args.filter_value).strip().lower()
            mask = ds.data[col].astype(str).str.strip().str.lower() == target_val
            keep_idx = [i for i, m in enumerate(mask.tolist()) if m]
            ds.paths = [ds.paths[i] for i in keep_idx]
            ds.targets = [ds.targets[i] for i in keep_idx]
            ds.samples = list(zip(ds.paths, ds.targets))
            ds.labels = [ds.idx_to_class[t] for t in ds.targets]
            ds.classes = sorted({ds.idx_to_class[t] for t in ds.targets})
            ds.data = ds.data.iloc[keep_idx].reset_index(drop=True)
            print(f"Applied filter: {args.filter_rank} == {args.filter_value}. Remaining samples: {len(ds.paths)}")
        else:
            print(f"[WARN] filter_rank '{args.filter_rank}' not found in dataset; skipping filter.")
    return ds


def evaluate_closed(args: argparse.Namespace, model, dataset, device: str) -> Dict[str, float]:
    templates = build_templates(args.template_style)
    classnames = list(dataset.classes)
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        weights = []
        for cname in tqdm(classnames, desc="Encoding prompts", leave=False):
            texts = [tpl(cname) for tpl in templates]
            toks = tokenizer(texts).to(device)
            feats = model.encode_text(toks)
            feats = F.normalize(feats, dim=-1).mean(dim=0)
            feats = feats / feats.norm()
            weights.append(feats)
        classifier = torch.stack(weights, dim=1).to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
    )

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    n = 0.0
    topk_keys = [1, min(3, len(classnames)), min(5, len(classnames))]
    topk = {k: 0.0 for k in topk_keys}
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Closed eval", leave=False):
            images = images.to(device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            targets = targets.to(device)
            with autocast():
                feats, _ = model.encode_image(images)
                feats = F.normalize(feats, dim=-1)
                logits = model.logit_scale.exp() * feats @ classifier
            acc = accuracy(logits, targets, topk=topk.keys())
            for k, v in acc.items():
                topk[k] += v
            n += images.size(0)

    if n == 0:
        return {"closed_top1": float("nan"), "closed_top3": float("nan"), "closed_top5": float("nan")}

    metrics = {}
    for k in topk_keys:
        metrics[f"closed_top{k}"] = topk[k] / n
    return metrics


def load_species_text(species_emb_npy: str, species_names_json: str, device: str, cast_dtype=None):
    emb = np.load(species_emb_npy)
    species_emb = torch.from_numpy(emb).to(device)
    if cast_dtype is not None:
        species_emb = species_emb.to(dtype=cast_dtype)
    with open(species_names_json, 'r') as f:
        names = json.load(f)
    return species_emb, names


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
                return " ".join(tax[:6] + [f"{tax[5]} {tax[6]}"])
            return " ".join(tax[:idx+1])

    label_per_species = [aggregated_label(e) for e in names]
    mapping = {}
    for i, lab in enumerate(label_per_species):
        mapping.setdefault(lab, []).append(i)

    allowed = set(allowed_labels)
    mapping = {lab: idxs for lab, idxs in mapping.items() if lab in allowed}
    label_order = [lab for lab in allowed_labels if lab in mapping]
    return mapping, label_order


def filter_and_remap_dataset(ds, label_order: List[str]):
    new_c2i = {c: i for i, c in enumerate(label_order)}
    new_targets = []
    new_paths = []
    for path, old_idx in ds.samples:
        cls_name = ds.idx_to_class[old_idx]
        if cls_name in new_c2i:
            new_paths.append(path)
            new_targets.append(new_c2i[cls_name])
    ds.class_to_idx = new_c2i
    ds.idx_to_class = {i: c for c, i in new_c2i.items()}
    ds.classes = label_order
    ds.paths = new_paths
    ds.targets = new_targets
    ds.samples = list(zip(ds.paths, ds.targets))
    return ds


def evaluate_open(args: argparse.Namespace, model, dataset, device: str) -> Dict[str, float]:
    cast_dtype = get_cast_dtype(args.precision)
    species_emb, names = load_species_text(args.species_emb_npy, args.species_names_json, device=device, cast_dtype=cast_dtype)

    if args.restrict_hf_rank and args.restrict_hf_value and args.hard_restrict_hf:
        level_map = {"kingdom":0,"phylum":1,"cls":2,"class":2,"order":3,"family":4,"genus":5,"species":6}
        r_idx = level_map[args.restrict_hf_rank.lower()]
        target_val = str(args.restrict_hf_value).lower()
        keep = [i for i, entry in enumerate(names) if str(entry[0][r_idx]).lower() == target_val]
        species_emb = species_emb[:, keep]
        names = [names[i] for i in keep]

    mapping, label_order = build_mapping(names, args.rank, dataset.classes, args.rank_only)
    if args.restrict_hf_rank and args.restrict_hf_value and not args.hard_restrict_hf:
        level_map = {"kingdom":0,"phylum":1,"cls":2,"class":2,"order":3,"family":4,"genus":5,"species":6}
        r_idx = level_map[args.restrict_hf_rank.lower()]
        target_val = str(args.restrict_hf_value).lower()
        kept = []
        for lab in list(label_order):
            indices = mapping.get(lab, [])
            keep_lab = any(str(names[i][0][r_idx]).lower() == target_val for i in indices)
            if keep_lab:
                kept.append(lab)
        mapping = {lab: mapping[lab] for lab in kept}
        label_order = kept

    coverage = len(label_order)
    if coverage == 0:
        return {"open_top1": float("nan"), "open_top3": float("nan"), "open_top5": float("nan")}

    dataset = filter_and_remap_dataset(dataset, label_order)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
    )

    A = torch.zeros((species_emb.shape[1], len(label_order)), dtype=torch.float32, device=device)
    for j, lab in enumerate(label_order):
        idxs = mapping[lab]
        A[idxs, j] = 1.0

    autocast = get_autocast(args.precision)
    model.eval()
    n = 0
    topk_keys = [1, min(3, len(label_order)), min(5, len(label_order))]
    topk = {k: 0.0 for k in topk_keys}
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Open eval", leave=False):
            images = images.to(device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            targets = targets.to(device)
            with autocast():
                feats, _ = model.encode_image(images)
                feats = F.normalize(feats, dim=-1)
                logits_species = model.logit_scale.exp() * feats @ species_emb
                probs_species = F.softmax(logits_species, dim=-1)
                probs_rank = probs_species @ A
            acc = accuracy(probs_rank, targets, topk=topk.keys())
            for k, v in acc.items():
                topk[k] += v
            n += images.size(0)

    if n == 0:
        return {"open_top1": float("nan"), "open_top3": float("nan"), "open_top5": float("nan")}

    metrics = {}
    for k in topk_keys:
        metrics[f"open_top{k}"] = topk[k] / n
    return metrics


def main():
    args = parse_args()
    checkpoints = list_checkpoints(args.checkpoints_dir, args.max_checkpoints)
    if not checkpoints:
        raise RuntimeError("No checkpoints found in directory")

    results = []

    for ckpt in checkpoints:
        print(f"Evaluating {ckpt.name}...")
        model, preprocess_val = load_model(args)
        load_checkpoint(model, ckpt, args.device)

        row = {"checkpoint": ckpt.name}

        if args.eval_closed:
            ds_closed = load_dataset(args, preprocess_val, for_closed=True)
            metrics_closed = evaluate_closed(args, model, ds_closed, args.device)
            row.update(metrics_closed)

        if args.eval_open:
            ds_open = load_dataset(args, preprocess_val, for_closed=False)
            metrics_open = evaluate_open(args, model, ds_open, args.device)
            row.update(metrics_open)

        results.append(row)
        del model
        torch.cuda.empty_cache()

    csv_path = Path(args.output_csv)
    fieldnames = sorted({k for row in results for k in row.keys()})
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote results to {csv_path}")

    best_field = "open_top1" if args.eval_open else "closed_top1"
    best = max(results, key=lambda r: r.get(best_field, float('-inf')))
    print(f"Best checkpoint by {best_field}: {best['checkpoint']} ({best.get(best_field)})")


if __name__ == "__main__":
    main()
