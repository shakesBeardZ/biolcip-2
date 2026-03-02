from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from dataset_train import CoralTrainingDataset
from src.open_clip import create_model_and_transforms, get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export coral text embeddings aligned with the chosen checkpoint")
    parser.add_argument("--csv", required=True, help="Path to the master coral CSV")
    parser.add_argument("--split-column", default="split", help="Column indicating split membership")
    parser.add_argument(
        "--include-splits",
        default=None,
        help="Comma-separated list of split values to include (default: use all rows)",
    )
    parser.add_argument("--model", default="hf-hub:imageomics/bioclip-2", help="Model identifier")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load before encoding")
    parser.add_argument("--output-prefix", required=True, help="Prefix for output files (e.g., ./embeddings/coral_genus)")
    parser.add_argument(
        "--target-level",
        default="species",
        choices=["genus", "species"],
        help="Taxonomy level to treat as the class label",
    )
    parser.add_argument(
        "--caption-level",
        default=None,
        choices=["genus", "species"],
        help="Taxonomy level to use for captions (default: same as --target-level)",
    )
    parser.add_argument(
        "--caption-mode",
        default="chain",
        choices=["chain", "rank_only"],
        help="Caption composition mode (match fine-tune configuration)",
    )
    parser.add_argument("--rank-only", action="store_true", help="Use rank-only labels for the dataset")
    parser.add_argument("--apply-normalize-anthozoa", action="store_true", help="Apply Anthozoa normalization")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size when encoding text")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> CoralTrainingDataset:
    splits = None
    if args.include_splits:
        splits = [s.strip() for s in args.include_splits.split(",") if s.strip()]
    caption_level = args.caption_level or args.target_level
    dataset = CoralTrainingDataset(
        csv_file=args.csv,
        transform=None,
        target_level=args.target_level,
        rank_only=args.rank_only,
        caption_level=caption_level,
        caption_mode=args.caption_mode,
        normalize_anthozoa=args.apply_normalize_anthozoa,
        split_column=args.split_column if splits else None,
        include_splits=splits if splits else None,
        return_metadata=False,
    )
    return dataset


def gather_entries(dataset: CoralTrainingDataset, target_level: str) -> List[Tuple[str, List[str]]]:
    entries: Dict[str, Tuple[str, List[str]]] = OrderedDict()
    for idx, taxon in enumerate(dataset._taxa):
        genus = (taxon.genus or "").strip()
        species_ep = (taxon.species or "").strip()
        taxonomy = list(taxon.to_tuple())
        if target_level == "species":
            if not genus or not species_ep:
                continue
            key = f"{genus.capitalize()} {species_ep.lower()}"
            taxonomy = [taxonomy[0], taxonomy[1], taxonomy[2], taxonomy[3], taxonomy[4], genus.capitalize(), species_ep.lower()]
        else:
            if not genus:
                continue
            key = genus.capitalize()
            taxonomy = [taxonomy[0], taxonomy[1], taxonomy[2], taxonomy[3], taxonomy[4], genus.capitalize(), ""]
        if key in entries:
            continue
        text = dataset.captions[idx]
        entries[key] = (text, taxonomy)
    return [(text, taxonomy) for text, taxonomy in entries.values()]


def load_model_and_tokenizer(args: argparse.Namespace):
    model, _, _ = create_model_and_transforms(
        args.model,
        pretrained=None,
        precision="fp32",
        device=args.device,
        output_dict=True,
        load_weights_only=True,
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    tokenizer = get_tokenizer(args.model)
    model = model.to(args.device)
    model.eval()
    return model, tokenizer


def encode_texts(model, tokenizer, texts: List[str], device: str, batch_size: int) -> torch.Tensor:
    embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            text_features = model.encode_text(tokens, normalize=True)
            embeddings.append(text_features.detach().cpu())
    return torch.cat(embeddings, dim=0)


def save_embeddings(prefix: str, embeddings: torch.Tensor, names: List[List[str]]):
    out_dir = Path(prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = Path(prefix + "_emb.npy")
    json_path = Path(prefix + "_names.json")
    np.save(npy_path, embeddings.T.numpy())
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([[tax, ""] for tax in names], f)
    print(f"Wrote embeddings: {npy_path}")
    print(f"Wrote names:      {json_path}")


def main():
    args = parse_args()
    dataset = load_dataset(args)
    print(f"Loaded dataset with {len(dataset)} samples")
    entries = gather_entries(dataset, args.target_level)
    print(f"Unique {args.target_level} entries: {len(entries)}")
    if not entries:
        raise RuntimeError("No entries collected; check CSV and split settings")

    texts = [text for text, _ in entries]
    taxonomy_lists = [[str(x or "") for x in tax] for _, tax in entries]

    model, tokenizer = load_model_and_tokenizer(args)
    embeddings = encode_texts(model, tokenizer, texts, args.device, args.batch_size)

    save_embeddings(args.output_prefix, embeddings, taxonomy_lists)


if __name__ == "__main__":
    main()
