"""
Closed-domain evaluation over your CSV label set.

Scores only against your dataset's label space (no open-domain aggregation).
Supports taxonomy-aware labels via FranCatalogueLoader and prompt templates.
"""
import argparse
import json
import os
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .zero_shot_iid import accuracy
from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
)
from ..training.precision import get_autocast
from ..training.imagenet_zeroshot_data import openai_imagenet_template
from dataset_catalogue import FranCatalogueLoader
from dataset_rsg import RSGPatchDataset

DATASET_CHOICES = ["catalogue", "rsg"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--pretrained", type=str, default=None, help="Pretrained tag/path passed to create_model_and_transforms (e.g., openai, laion2b_s32b_b79k).")
    ap.add_argument("--dataset_loader", type=str, default="catalogue", choices=DATASET_CHOICES)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--rank", type=str, default="genus", choices=["kingdom","phylum","cls","order","family","genus","species","scientific"]) 
    ap.add_argument("--rank_only", action="store_true")
    ap.add_argument("--normalize_anthozoa", action="store_true")
    ap.add_argument("--require_species", action="store_true")
    ap.add_argument("--template_style", type=str, default="openai", choices=["openai","plain","bio"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32","amp","bf16","amp_bfloat16","amp_bf16"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint (.pt) to load before evaluation")
    ap.add_argument("--dump_predictions", type=str, default=None)
    ap.add_argument("--dump_predictions_csv", type=str, default=None, help="If provided, dump one CSV row per evaluated sample (true/pred labels, top-k labels/probs).")
    ap.add_argument("--dump_topk", type=int, default=5)
    ap.add_argument("--dump_misclassified_csv", type=str, default=None)
    ap.add_argument("--dump_metrics_json", type=str, default=None, help="If provided, dump aggregate metrics/top-k to a JSON file.")
    ap.add_argument("--class_list", type=str, default=None, help="Optional file containing taxonomy chains to define the class set")
    # optional dataset-side taxonomy filter (e.g., only Order=Scleractinia)
    ap.add_argument("--filter_rank", type=str, default=None, choices=["kingdom","phylum","cls","class","order","family","genus","species"])
    ap.add_argument("--filter_value", type=str, default=None)
    ap.add_argument("--rsg_taxonomy_csv", type=str, default=None)
    ap.add_argument("--rsg_corals_only", action="store_true")
    ap.add_argument("--rsg_path_key", type=str, default=None)
    ap.add_argument("--rsg_label_key", type=str, default=None)
    return ap.parse_args()


def label_from_taxonomy(tax: List[str], rank: str, rank_only: bool) -> str:
    level_map = {"kingdom": 0, "phylum": 1, "cls": 2, "order": 3, "family": 4, "genus": 5, "species": 6, "scientific": 6}
    rank = rank.lower()
    idx = level_map.get(rank, 5)
    # pad taxonomy list to at least 7 elements
    padded = (tax + [""] * 7)[:7]
    if rank_only:
        if rank == "species":
            genus = padded[5].strip()
            species = padded[6].strip()
            return f"{genus} {species}".strip()
        if rank == "scientific":
            genus = padded[5].strip()
            species = padded[6].strip()
            return f"{genus} {species}".strip()
        return padded[idx].strip()
    else:
        if rank == "species" or rank == "scientific":
            genus = padded[5].strip()
            species = padded[6].strip()
            chain = [p.strip() for p in padded[:5] if p.strip()]
            if genus or species:
                chain.append(f"{genus} {species}".strip())
            return " ".join(chain).strip()
        chain = [p.strip() for p in padded[: idx + 1] if p.strip()]
        return " ".join(chain).strip()


def apply_class_list(ds, class_file: str, rank: str, rank_only: bool):
    with open(class_file, 'r', encoding='utf-8') as f:
        raw = [line.strip() for line in f if line.strip()]
    desired_labels = []
    for line in raw:
        parts = [p.strip() for p in line.split(',')]
        if not parts:
            continue
        desired_labels.append(label_from_taxonomy(parts, rank, rank_only))
    desired_labels = [lbl for lbl in desired_labels if lbl]
    label_to_idx = {lbl: i for i, lbl in enumerate(desired_labels)}
    new_paths, new_targets = [], []
    seen = set()
    for path, idx in ds.samples:
        label = ds.idx_to_class[idx]
        if label in label_to_idx:
            new_paths.append(path)
            new_targets.append(label_to_idx[label])
            seen.add(label)
    missing = [lbl for lbl in desired_labels if lbl not in seen]
    if missing:
        print(f"[WARN] class_list missing labels present in file (no samples): {missing}")
    if not new_paths:
        print("[WARN] No samples matched the provided class list.")
    ds.paths = new_paths
    ds.targets = new_targets
    ds.samples = list(zip(ds.paths, ds.targets))
    ds.classes = desired_labels
    ds.class_to_idx = label_to_idx
    ds.idx_to_class = {i: lbl for lbl, i in label_to_idx.items()}
    ds.labels = [ds.idx_to_class[t] for t in ds.targets]
    if hasattr(ds, 'data') and len(ds.paths) < len(ds.data):
        cols = {c.lower(): c for c in ds.data.columns}
        path_col = cols.get('path')
        if path_col:
            path_set = {str(p) for p in ds.paths}
            ds.data = ds.data[ds.data[path_col].astype(str).isin(path_set)].reset_index(drop=True)
    return ds


def build_templates(style: str):
    if style == "plain":
        return [lambda c: f"{c}"]
    elif style == "bio":
        return [
            lambda c: f"{c}",
            lambda c: f"a photo of {c}",
            lambda c: f"an image of {c}",
            lambda c: f"a biological photograph of {c}",
        ]
    else:
        return openai_imagenet_template


def zeroshot_classifier(model_name: str, model, classnames: List[str], templates, device):
    tokenizer = get_tokenizer(model_name)
    with torch.no_grad():
        ws = []
        for cname in tqdm(classnames):
            texts = [t(cname) for t in templates]
            toks = tokenizer(texts).to(device)
            feats = model.encode_text(toks)
            feats = F.normalize(feats, dim=-1).mean(dim=0)
            feats = feats / feats.norm()
            ws.append(feats)
        W = torch.stack(ws, dim=1).to(device)
    return W


def ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
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

    taxonomy_filters = None
    if args.filter_rank and args.filter_value and args.dataset_loader == "catalogue":
        rank_key = args.filter_rank.lower()
        if rank_key == 'class':
            rank_key = 'cls'
        taxonomy_filters = {rank_key: args.filter_value}

    if args.dataset_loader == "rsg":
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
            require_species=False,
            rank_only=args.rank_only,
            normalize_anthozoa=args.normalize_anthozoa,
            corals_only=args.rsg_corals_only,
            **rsg_kwargs,
        )
    else:
        ds = FranCatalogueLoader(
            csv_file=args.csv,
            data_root=args.data_root,
            taxonomic_level=args.rank,
            transform=preprocess_val,
            require_species=args.require_species,
            rank_only=args.rank_only,
            normalize_anthozoa=args.normalize_anthozoa,
            taxonomy_filters=taxonomy_filters,
        )

    filter_applied_in_loader = args.dataset_loader == 'catalogue' and taxonomy_filters is not None
    if args.filter_rank and args.filter_value and not filter_applied_in_loader:
        rank_key = args.filter_rank.lower()
        if rank_key == "class":
            rank_key = "cls"
        if hasattr(ds, "data") and rank_key in {c.lower() for c in ds.data.columns}:
            cols = {c.lower(): c for c in ds.data.columns}
            col = cols[rank_key]
            target_val = str(args.filter_value).strip().lower()
            mask = ds.data[col].astype(str).str.strip().str.lower() == target_val
            keep_idx = [i for i, m in enumerate(mask.tolist()) if m]
            ds.paths = [ds.paths[i] for i in keep_idx]
            ds.targets = [ds.targets[i] for i in keep_idx]
            ds.samples = list(zip(ds.paths, ds.targets))
            ds.labels = [ds.idx_to_class[t] for t in ds.targets]
            ds.classes = sorted({ds.idx_to_class[t] for t in ds.targets})
            if hasattr(ds, "data"):
                ds.data = ds.data.iloc[keep_idx].reset_index(drop=True)
            print(f"Applied filter: {args.filter_rank} == {args.filter_value}. Remaining samples: {len(ds.paths)}")
        else:
            print(f"[WARN] filter_rank '{args.filter_rank}' not found in dataset columns; skipping filter.")

    if args.class_list:
        ds = apply_class_list(ds, args.class_list, args.rank, args.rank_only)

    templates = build_templates(args.template_style)
    classnames = list(ds.classes)
    classifier = zeroshot_classifier(args.model, model, classnames, templates, device)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=False,
        pin_memory=False,
    )
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    dump = args.dump_predictions is not None
    dump_rows = []
    dump_csv = args.dump_predictions_csv is not None
    csv_rows = []
    collect_mis = args.dump_misclassified_csv is not None

    n = 0.0
    topk = {1: 0.0, min(3, len(classnames)): 0.0, min(5, len(classnames)): 0.0}
    model.eval()
    global_index = 0
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            targets = targets.to(device)
            with autocast():
                img_feats, _ = model.encode_image(images)
                img_feats = F.normalize(img_feats, dim=-1)
                logits = model.logit_scale.exp() * img_feats @ classifier

            acc = accuracy(logits, targets, topk=topk.keys())
            for k, v in acc.items():
                topk[k] += v
            n += images.size(0)

            probs = F.softmax(logits, dim=-1)
            bs = images.size(0)
            for bi in range(bs):
                pv, pi = torch.topk(probs[bi], k=min(args.dump_topk, probs.shape[-1]))
                topk_preds = [
                    {
                        "rank": j + 1,
                        "index": int(pi[j].item()),
                        "label": classnames[int(pi[j].item())],
                        "prob": float(pv[j].item()),
                    }
                    for j in range(pi.numel())
                ]
                true_idx = int(targets[bi].item())
                true_label = classnames[true_idx]
                pred1_idx = int(pi[0].item())
                pred1_label = classnames[pred1_idx]
                pred1_prob = float(pv[0].item())
                path = ds.paths[global_index + bi]
                row = {
                    "sample_index": int(global_index + bi),
                    "path": path,
                    "true_index": true_idx,
                    "true_label": true_label,
                    "pred_index": pred1_idx,
                    "pred_label": pred1_label,
                    "pred_prob": pred1_prob,
                    "correct_top1": int(pred1_idx == true_idx),
                }
                # Flatten top-k columns for downstream tabular metrics.
                for entry in topk_preds:
                    rk = entry["rank"]
                    row[f"top{rk}_index"] = entry["index"]
                    row[f"top{rk}_label"] = entry["label"]
                    row[f"top{rk}_prob"] = entry["prob"]

                if dump:
                    dump_rows.append({
                        "sample_index": int(global_index + bi),
                        "path": path,
                        "true_index": true_idx,
                        "true_label": true_label,
                        "pred_index": pred1_idx,
                        "pred_label": pred1_label,
                        "pred_prob": pred1_prob,
                        "topk": topk_preds,
                    })
                if dump_csv or collect_mis:
                    csv_rows.append(row)
            global_index += bs

    if n == 0:
        print("No samples remained after filtering; skipping accuracy computation.")
        return

    for k in list(topk.keys()):
        topk[k] /= n

    print("Results:")
    for k in sorted(topk.keys()):
        print(f"  top{k}: {topk[k]*100:.2f}")

    metrics_payload = {
        "model": args.model,
        "pretrained": args.pretrained,
        "checkpoint": args.checkpoint,
        "rank": args.rank,
        "template_style": args.template_style,
        "n_samples": int(n),
        "num_classes": int(len(classnames)),
    }
    for k in sorted(topk.keys()):
        metrics_payload[f"top{k}"] = float(topk[k])

    if dump:
        ensure_parent_dir(args.dump_predictions)
        with open(args.dump_predictions, 'w') as f:
            json.dump(dump_rows, f, indent=2)
        print("Wrote:", args.dump_predictions)

    if dump_csv:
        import csv as csv_mod
        ensure_parent_dir(args.dump_predictions_csv)
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
        else:
            fieldnames = [
                "sample_index", "path", "true_index", "true_label",
                "pred_index", "pred_label", "pred_prob", "correct_top1",
            ]
        with open(args.dump_predictions_csv, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print("Wrote:", args.dump_predictions_csv)

    if collect_mis:
        import csv
        ensure_parent_dir(args.dump_misclassified_csv)
        with open(args.dump_misclassified_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["path", "pred_label", "true_label"])
            for row in csv_rows:
                if int(row["correct_top1"]) == 0:
                    w.writerow([row["path"], row["pred_label"], row["true_label"]])
        print("Wrote:", args.dump_misclassified_csv)

    if args.dump_metrics_json:
        ensure_parent_dir(args.dump_metrics_json)
        with open(args.dump_metrics_json, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        print("Wrote:", args.dump_metrics_json)


if __name__ == "__main__":
    main()
