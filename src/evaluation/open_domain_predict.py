"""
Open-domain taxonomy prediction (demo-like) for a single image or a folder.

Scores an image against precomputed species text embeddings (from
imageomics/TreeOfLife-200M), then:
 - Species: returns top-k species taxonomy strings (optionally with common names)
 - Higher rank: aggregates species probabilities to the chosen rank and returns top-k

Usage example:

python -m src.evaluation.open_domain_predict \
  --model hf-hub:imageomics/bioclip-2 \
  --image "/path/to/image.jpg" \
  --rank genus \
  --species_emb_npy data/species_embeddings/embeddings/txt_emb_species.npy \
  --species_names_json data/species_embeddings/embeddings/txt_emb_species.json \
  --topk 5

"""
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..open_clip import create_model_and_transforms
from ..training.precision import get_autocast


RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def load_species_text(species_emb_npy: str, species_names_json: str, device: torch.device, cast_dtype=None) -> Tuple[torch.Tensor, list]:
    emb = np.load(species_emb_npy)
    species_emb = torch.from_numpy(emb).to(device)
    if cast_dtype is not None:
        species_emb = species_emb.to(dtype=cast_dtype)
    with open(species_names_json, 'r') as f:
        names = json.load(f)
    return species_emb, names


def format_name(taxon: List[str], common: str) -> str:
    taxon_str = " ".join(taxon)
    if common:
        return f"{taxon_str} ({common})"
    return taxon_str


def aggregate_probs_to_rank(probs_species: torch.Tensor, names: list, rank_idx: int, topk: int = 5) -> List[Tuple[str, float]]:
    # names: [[ [Kingdom, Phylum, Class, Order, Family, Genus, Species], common_name ], ...]
    agg = {}
    for i, p in enumerate(probs_species.tolist()):
        taxon = names[i][0]
        key = " ".join(taxon[: rank_idx + 1])
        agg[key] = agg.get(key, 0.0) + p
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return items


def topk_species(probs_species: torch.Tensor, names: list, topk: int = 5) -> List[Tuple[str, float]]:
    vals, idxs = torch.topk(probs_species, k=min(topk, probs_species.numel()))
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        taxon, common = names[i]
        out.append((format_name(taxon, common), v))
    return out


def predict_image(model, preprocess_val, image_path: str, species_emb: torch.Tensor, names: list, rank: str, device: torch.device, precision: str, topk: int = 5) -> List[Tuple[str, float]]:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img_t = preprocess_val(img).unsqueeze(0).to(device)
    autocast = get_autocast(precision)
    with torch.no_grad():
        with autocast():
            feats, _ = model.encode_image(img_t)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            logits = model.logit_scale.exp() * feats @ species_emb
            probs = torch.nn.functional.softmax(logits.squeeze(0), dim=-1)

    if rank.lower() == "species":
        return topk_species(probs, names, topk)
    else:
        rank_idx = RANKS.index(rank.lower())
        return aggregate_probs_to_rank(probs, names, rank_idx, topk)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--image", type=str, required=True, help="Path to an image or a directory of images")
    ap.add_argument("--rank", type=str, default="species", choices=[r.title() for r in RANKS] + list(RANKS))
    ap.add_argument("--species_emb_npy", type=str, required=True)
    ap.add_argument("--species_names_json", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "amp", "bf16", "amp_bfloat16", "amp_bf16"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rank = args.rank.lower()

    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=None,
        precision=args.precision,
        device=args.device,
        output_dict=True,
        load_weights_only=True,
    )

    cast_dtype = None
    if args.precision in ("fp16", "pure_fp16"):
        cast_dtype = torch.float16
    elif args.precision in ("bf16", "pure_bf16", "amp_bfloat16", "amp_bf16"):
        cast_dtype = torch.bfloat16

    species_emb, names = load_species_text(args.species_emb_npy, args.species_names_json, device=torch.device(args.device), cast_dtype=cast_dtype)

    paths = []
    if os.path.isdir(args.image):
        for nm in os.listdir(args.image):
            if nm.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(args.image, nm))
    else:
        paths.append(args.image)

    for p in paths:
        results = predict_image(model, preprocess_val, p, species_emb, names, rank, torch.device(args.device), args.precision, topk=args.topk)
        print("Image:", p)
        for name, prob in results:
            print(f"  {name}\t{prob:.4f}")
        print()

