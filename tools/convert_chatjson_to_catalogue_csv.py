#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


CANONICAL_CORAL_GENERA = [
    "Porites",
    "Acropora",
    "Pocillopora",
    "Montipora",
    "Goniastrea",
    "Echinopora",
    "Stylophora",
    "Favites",
    "Lobophyllia",
    "Seriatopora",
    "Galaxea",
    "Astreopora",
    "Tubastraea",
    "Plerogyra",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert chat-style JSON annotations (id/images/messages) into a CSV compatible "
            "with FranCatalogueLoader / closed_domain_eval."
        )
    )
    parser.add_argument("--input-json", required=True, help="Path to JSON file.")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV.")
    parser.add_argument(
        "--taxonomy-csv",
        default=None,
        help=(
            "Optional CSV containing taxonomy columns (kingdom/phylum/class/order/family/genus/species) "
            "used to populate full chains by genus."
        ),
    )
    parser.add_argument(
        "--path-replace-from",
        default="",
        help="Optional substring/prefix to replace in image paths.",
    )
    parser.add_argument(
        "--path-replace-to",
        default="",
        help="Replacement for --path-replace-from.",
    )
    parser.add_argument(
        "--split-value",
        default="test",
        help="Value to write into Split column (default: test).",
    )
    parser.add_argument(
        "--source-name",
        default="",
        help="Optional fixed value for CoralNet_source. If empty, inferred from path.",
    )
    parser.add_argument(
        "--filter-known-genera",
        action="store_true",
        help=f"Keep only known coral genera ({len(CANONICAL_CORAL_GENERA)}-class list).",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="If set, drop rows whose mapped path does not exist.",
    )
    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="Print skipped rows with reason.",
    )
    return parser.parse_args()


def clean_str(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def normalize_genus_label(raw: str, canonical_map: Dict[str, str]) -> str:
    text = clean_str(raw)
    if not text:
        return ""

    # Keep only first line; labels should be one genus.
    text = text.splitlines()[0].strip()
    # Try to extract first alphabetic token.
    m = re.search(r"[A-Za-z][A-Za-z-]*", text)
    genus = m.group(0) if m else text
    genus = genus.strip(".,:;!?()[]{}<>\"'`").strip()
    if not genus:
        return ""

    genus_key = genus.lower()
    if genus_key in canonical_map:
        return canonical_map[genus_key]
    return genus[:1].upper() + genus[1:].lower()


def extract_assistant_label(messages) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and str(msg.get("role", "")).strip().lower() == "assistant":
            return clean_str(msg.get("content", ""))
    return ""


def map_path(path_str: str, old: str, new: str) -> str:
    path_str = clean_str(path_str)
    if old:
        return path_str.replace(old, new, 1)
    return path_str


def infer_source(path_str: str, source_name: str) -> str:
    if source_name:
        return source_name
    p = Path(path_str)
    if p.parent.name and p.parent.parent.name:
        return p.parent.parent.name
    if p.parent.name:
        return p.parent.name
    return "unknown"


def find_ci_col(columns, candidates) -> Optional[str]:
    col_map = {c.lower(): c for c in columns}
    for c in candidates:
        hit = col_map.get(c.lower())
        if hit:
            return hit
    return None


def build_taxonomy_lookup(taxonomy_csv: Optional[str]) -> Dict[str, dict]:
    if not taxonomy_csv:
        return {}

    df = pd.read_csv(taxonomy_csv)
    if df.empty:
        return {}

    genus_col = find_ci_col(df.columns, ["genus"])
    if not genus_col:
        raise KeyError("taxonomy_csv does not contain a genus column.")

    kingdom_col = find_ci_col(df.columns, ["kingdom"])
    phylum_col = find_ci_col(df.columns, ["phylum"])
    class_col = find_ci_col(df.columns, ["class", "cls"])
    subclass_col = find_ci_col(df.columns, ["subclass"])
    order_col = find_ci_col(df.columns, ["order"])
    family_col = find_ci_col(df.columns, ["family"])
    species_col = find_ci_col(df.columns, ["species"])

    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        genus = clean_str(row.get(genus_col))
        if not genus:
            continue
        key = genus.lower()
        if key not in out:
            out[key] = {
                "kingdom": "",
                "phylum": "",
                "class": "",
                "subclass": "",
                "order": "",
                "family": "",
                "genus": genus,
                "species": "",
            }

        rec = out[key]
        if kingdom_col:
            rec["kingdom"] = rec["kingdom"] or clean_str(row.get(kingdom_col))
        if phylum_col:
            rec["phylum"] = rec["phylum"] or clean_str(row.get(phylum_col))
        if class_col:
            rec["class"] = rec["class"] or clean_str(row.get(class_col))
        if subclass_col:
            rec["subclass"] = rec["subclass"] or clean_str(row.get(subclass_col))
        if order_col:
            rec["order"] = rec["order"] or clean_str(row.get(order_col))
        if family_col:
            rec["family"] = rec["family"] or clean_str(row.get(family_col))
        if species_col:
            rec["species"] = rec["species"] or clean_str(row.get(species_col))

    return out


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    canonical_map = {g.lower(): g for g in CANONICAL_CORAL_GENERA}
    taxonomy_lookup = build_taxonomy_lookup(args.taxonomy_csv)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    rows = []
    skipped_missing_image = 0
    skipped_missing_label = 0
    skipped_unknown_genus = 0
    skipped_missing_file = 0

    for item in data:
        item_id = clean_str(item.get("id", ""))
        images = item.get("images", [])
        if not images:
            skipped_missing_image += 1
            if args.verbose_skips:
                print(f"[skip] id={item_id}: no images")
            continue

        orig_path = clean_str(images[0])
        if not orig_path:
            skipped_missing_image += 1
            if args.verbose_skips:
                print(f"[skip] id={item_id}: empty image path")
            continue

        mapped_path = map_path(orig_path, args.path_replace_from, args.path_replace_to)
        if args.check_exists and not Path(mapped_path).exists():
            skipped_missing_file += 1
            if args.verbose_skips:
                print(f"[skip] id={item_id}: missing file after path mapping: {mapped_path}")
            continue

        assistant_raw = extract_assistant_label(item.get("messages", []))
        genus = normalize_genus_label(assistant_raw, canonical_map)
        if not genus:
            skipped_missing_label += 1
            if args.verbose_skips:
                print(f"[skip] id={item_id}: missing assistant genus label")
            continue

        if args.filter_known_genera and genus.lower() not in canonical_map:
            skipped_unknown_genus += 1
            if args.verbose_skips:
                print(f"[skip] id={item_id}: unknown genus '{genus}'")
            continue

        tax = taxonomy_lookup.get(genus.lower(), {})
        row = {
            "id": item_id,
            "CoralNet_source": infer_source(mapped_path, args.source_name),
            "Image_name": Path(mapped_path).name,
            "patch_name": Path(mapped_path).name,
            "patch_path": mapped_path,
            "Path": mapped_path,
            "Split": args.split_value,
            "Experimental_label": genus,
            "Our_Labels": genus,
            "Taxonomic.rank.CoralNet.label": "Genus",
            "kingdom": tax.get("kingdom", "") or "Animalia",
            "phylum": tax.get("phylum", "") or "Cnidaria",
            "class": tax.get("class", "") or "Anthozoa",
            "subclass": tax.get("subclass", "") or "Hexacorallia",
            "order": tax.get("order", "") or "Scleractinia",
            "family": tax.get("family", ""),
            "genus": genus,
            "species": tax.get("species", ""),
            "assistant_content": assistant_raw,
            "image_path_original": orig_path,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    unique_genera = sorted(df["genus"].dropna().astype(str).str.strip().unique().tolist()) if not df.empty else []

    print(f"Wrote: {output_csv}")
    print(f"Rows written: {len(df)}")
    print(f"Unique genera ({len(unique_genera)}): {unique_genera}")
    print(f"Skipped (missing image): {skipped_missing_image}")
    print(f"Skipped (missing label): {skipped_missing_label}")
    print(f"Skipped (unknown genus): {skipped_unknown_genus}")
    print(f"Skipped (missing file): {skipped_missing_file}")


if __name__ == "__main__":
    main()
