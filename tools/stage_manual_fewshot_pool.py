#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_train import CoralTrainingDataset

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a manual few-shot inspection pool into per-class folders from a coral CSV. "
            "By default it uses split=train,val and genus chain labels."
        )
    )
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for staged pool.")
    parser.add_argument("--data-root", default=None, help="Optional root prepended to relative paths.")
    parser.add_argument("--path-key", default="patch_path", help="CSV path column (case-insensitive).")
    parser.add_argument(
        "--split-column",
        default="Split",
        help="Split column to filter rows before staging (case-insensitive).",
    )
    parser.add_argument(
        "--include-splits",
        default="train,val",
        help="Comma-separated splits to include (default: train,val).",
    )
    parser.add_argument(
        "--target-level",
        default="genus",
        choices=["kingdom", "phylum", "cls", "order", "family", "genus", "species", "scientific"],
        help="Taxonomic target level for class folders.",
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Use rank-only labels instead of chain-to-rank labels.",
    )
    parser.add_argument(
        "--caption-level",
        default=None,
        help="Caption level passed to dataset loader (defaults to target-level).",
    )
    parser.add_argument(
        "--caption-mode",
        default="chain",
        choices=["chain", "rank_only"],
        help="Caption mode passed to dataset loader.",
    )
    parser.add_argument(
        "--keep-missing-targets",
        action="store_true",
        help="Keep rows where requested target level is missing (fallback to parent rank).",
    )
    parser.add_argument(
        "--no-infer-genus-from-species",
        action="store_true",
        help="Keep genus empty when genus is missing in CSV (no inference from species binomial).",
    )
    parser.add_argument(
        "--no-normalize-anthozoa",
        action="store_true",
        help="Disable Hexacorallia->Anthozoa normalization under Cnidaria.",
    )
    parser.add_argument("--path-replace-from", default=None, help="Path substring/prefix to replace.")
    parser.add_argument("--path-replace-to", default=None, help="Replacement for --path-replace-from.")
    parser.add_argument(
        "--stage-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to stage files into output-dir/images (default: symlink).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap of staged images per class.",
    )
    parser.add_argument(
        "--diversity-image-column",
        default="Image_name",
        help="Column used to diversify picks within each class (default: Image_name).",
    )
    parser.add_argument(
        "--diversity-source-column",
        default="CoralNet_source",
        help="Second column used to diversify picks within each class (default: CoralNet_source).",
    )
    parser.add_argument(
        "--disable-diversity-selection",
        action="store_true",
        help="Disable diversity-aware selection and use plain first-N per class.",
    )
    parser.add_argument(
        "--exclude-file",
        default=None,
        help=(
            "Optional txt/csv file of rejected items to exclude. "
            "Accepted fields include _pool_dataset_index/dataset_index or "
            "_pool_source_path/src_path/path."
        ),
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete output-dir/images before staging (recommended for re-runs).",
    )
    parser.add_argument(
        "--shuffle-per-class",
        action="store_true",
        help="Shuffle samples within each class before applying max-per-class.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --shuffle-per-class is set.")
    parser.add_argument("--workers", type=int, default=24, help="Parallel workers for staging.")
    parser.add_argument(
        "--check-image-decode",
        action="store_true",
        help="Verify image decode (PIL convert RGB) before staging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination file/symlink if it exists.",
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=None,
        help="Optional global limit for quick testing.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    return parser.parse_args()


def sanitize_label(value: str, max_len: int = 90) -> str:
    text = re.sub(r"\s+", "_", value.strip())
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "EMPTY"
    return text[:max_len]


def split_values(arg_value: str) -> list[str]:
    vals = [s.strip() for s in arg_value.split(",") if s.strip()]
    if not vals:
        raise ValueError("No values parsed from --include-splits.")
    return vals


def normalize_path_text(value: str) -> str:
    return os.path.normpath(str(value).strip())


def normalize_meta_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _maybe_int(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return None


def load_exclusions(exclude_file: Optional[str]) -> tuple[set[int], set[str]]:
    if not exclude_file:
        return set(), set()

    p = Path(exclude_file)
    if not p.exists():
        raise FileNotFoundError(f"Exclude file not found: {exclude_file}")

    excluded_indices: set[int] = set()
    excluded_paths: set[str] = set()
    idx_candidates = ["_pool_dataset_index", "dataset_index", "index"]
    path_candidates = [
        "_pool_source_path",
        "src_path",
        "path",
        "_resolved_path",
        "Path",
        "patch_path",
    ]

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p, low_memory=False)
        cols = {c.lower(): c for c in df.columns}

        idx_col = None
        for c in idx_candidates:
            if c.lower() in cols:
                idx_col = cols[c.lower()]
                break
        path_col = None
        for c in path_candidates:
            if c.lower() in cols:
                path_col = cols[c.lower()]
                break

        if idx_col is not None:
            for v in df[idx_col].tolist():
                iv = _maybe_int(v)
                if iv is not None:
                    excluded_indices.add(iv)

        if path_col is not None:
            for v in df[path_col].tolist():
                if v is None or pd.isna(v):
                    continue
                text = str(v).strip()
                if text:
                    excluded_paths.add(normalize_path_text(text))
    else:
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                iv = _maybe_int(line)
                if iv is not None:
                    excluded_indices.add(iv)
                else:
                    excluded_paths.add(normalize_path_text(line))

    return excluded_indices, excluded_paths


def select_diverse_indices(
    indices: list[int],
    max_count: int,
    image_values: Optional[list[str]],
    source_values: Optional[list[str]],
) -> tuple[list[int], int, int]:
    remaining = list(indices)
    selected: list[int] = []
    seen_images: set[str] = set()
    seen_sources: set[str] = set()

    def get_image(i: int) -> str:
        if image_values is None:
            return ""
        return image_values[i]

    def get_source(i: int) -> str:
        if source_values is None:
            return ""
        return source_values[i]

    def consume(predicate) -> None:
        nonlocal remaining
        kept: list[int] = []
        for i in remaining:
            if len(selected) >= max_count:
                kept.append(i)
                continue
            img = get_image(i)
            src = get_source(i)
            if predicate(img, src):
                selected.append(i)
                if img:
                    seen_images.add(img)
                if src:
                    seen_sources.add(src)
            else:
                kept.append(i)
        remaining = kept

    has_image = image_values is not None
    has_source = source_values is not None

    if has_image and has_source:
        consume(lambda img, src: (not img or img not in seen_images) and (not src or src not in seen_sources))
        consume(lambda img, src: (not img or img not in seen_images))
        consume(lambda img, src: (not src or src not in seen_sources))
    elif has_image:
        consume(lambda img, src: (not img or img not in seen_images))
    elif has_source:
        consume(lambda img, src: (not src or src not in seen_sources))

    consume(lambda img, src: True)

    if len(selected) > max_count:
        selected = selected[:max_count]

    uniq_img = len({get_image(i) for i in selected if get_image(i)})
    uniq_src = len({get_source(i) for i in selected if get_source(i)})
    return selected, uniq_img, uniq_src


def find_column_case_insensitive(columns: list[str], name: str) -> Optional[str]:
    target = name.lower()
    for col in columns:
        if col.lower() == target:
            return col
    return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def do_stage(src: str, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir():
            raise IsADirectoryError(f"Destination is a directory: {dst}")
        dst.unlink()

    ensure_parent(dst)
    if mode == "symlink":
        os.symlink(src, str(dst))
    elif mode == "hardlink":
        os.link(src, str(dst))
    else:
        shutil.copy2(src, str(dst))


def verify_decode(path: str) -> None:
    if Image is None:
        raise RuntimeError("PIL is not available but --check-image-decode was requested.")
    with Image.open(path) as img:
        _ = img.convert("RGB")


@dataclass
class StageTask:
    dataset_index: int
    class_index: int
    class_label: str
    class_folder: str
    src_path: str
    dst_path: Path


def stage_one(task: StageTask, check_decode: bool, stage_mode: str, overwrite: bool) -> dict:
    if not task.src_path:
        return {
            "ok": False,
            "dataset_index": task.dataset_index,
            "class_index": task.class_index,
            "class_label": task.class_label,
            "class_folder": task.class_folder,
            "src_path": task.src_path,
            "dst_path": str(task.dst_path),
            "reason": "empty_path",
        }
    if not os.path.isfile(task.src_path):
        return {
            "ok": False,
            "dataset_index": task.dataset_index,
            "class_index": task.class_index,
            "class_label": task.class_label,
            "class_folder": task.class_folder,
            "src_path": task.src_path,
            "dst_path": str(task.dst_path),
            "reason": "missing",
        }
    try:
        if check_decode:
            verify_decode(task.src_path)
        do_stage(task.src_path, task.dst_path, mode=stage_mode, overwrite=overwrite)
        return {
            "ok": True,
            "dataset_index": task.dataset_index,
            "class_index": task.class_index,
            "class_label": task.class_label,
            "class_folder": task.class_folder,
            "src_path": task.src_path,
            "dst_path": str(task.dst_path),
            "reason": "ok",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "dataset_index": task.dataset_index,
            "class_index": task.class_index,
            "class_label": task.class_label,
            "class_folder": task.class_folder,
            "src_path": task.src_path,
            "dst_path": str(task.dst_path),
            "reason": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    include_splits = split_values(args.include_splits)
    excluded_indices, excluded_paths = load_exclusions(args.exclude_file)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_output and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = CoralTrainingDataset(
        csv_file=args.csv,
        data_root=args.data_root,
        target_level=args.target_level,
        rank_only=args.rank_only,
        caption_level=args.caption_level,
        caption_mode=args.caption_mode,
        normalize_anthozoa=not args.no_normalize_anthozoa,
        path_key=args.path_key,
        drop_missing_targets=not args.keep_missing_targets,
        split_column=args.split_column,
        include_splits=include_splits,
        infer_genus_from_species=not args.no_infer_genus_from_species,
        path_replace_from=args.path_replace_from,
        path_replace_to=args.path_replace_to,
        return_metadata=False,
    )

    print(
        f"Loaded dataset: samples={len(ds)}, classes={len(ds.classes)}, "
        f"target_level={args.target_level}, rank_only={args.rank_only}"
    )
    if excluded_indices or excluded_paths:
        print(
            f"Loaded exclusions: indices={len(excluded_indices)}, paths={len(excluded_paths)}"
        )

    image_col = None
    source_col = None
    image_values: Optional[list[str]] = None
    source_values: Optional[list[str]] = None
    if not args.disable_diversity_selection:
        image_col = find_column_case_insensitive(ds._data.columns.tolist(), args.diversity_image_column)
        source_col = find_column_case_insensitive(ds._data.columns.tolist(), args.diversity_source_column)
        if image_col is None:
            print(f"[WARN] Diversity image column not found: {args.diversity_image_column}")
        else:
            image_values = [normalize_meta_value(v) for v in ds._data[image_col].tolist()]
        if source_col is None:
            print(f"[WARN] Diversity source column not found: {args.diversity_source_column}")
        else:
            source_values = [normalize_meta_value(v) for v in ds._data[source_col].tolist()]
        if image_values is not None or source_values is not None:
            print(
                "Diversity selection enabled with columns: "
                f"image={image_col or 'N/A'}, source={source_col or 'N/A'}"
            )

    # Build class -> dataset indices.
    class_to_indices: dict[int, list[int]] = {}
    for idx, class_idx in enumerate(ds.target_indices):
        class_to_indices.setdefault(class_idx, []).append(idx)

    # Stable class folder names.
    class_folder_by_idx: dict[int, str] = {}
    for class_idx, class_name in ds.idx_to_class.items():
        class_folder_by_idx[class_idx] = f"class_{class_idx:03d}__{sanitize_label(class_name)}"

    # Choose staged indices per class.
    selected_indices: list[int] = []
    per_class_staged: dict[int, int] = {}
    per_class_available_after_exclusions: dict[int, int] = {}
    per_class_available_unique_images: dict[int, int] = {}
    per_class_available_unique_sources: dict[int, int] = {}
    per_class_unique_images: dict[int, int] = {}
    per_class_unique_sources: dict[int, int] = {}
    total_excluded_by_index = 0
    total_excluded_by_path = 0
    for class_idx in sorted(class_to_indices):
        indices = list(class_to_indices[class_idx])
        if excluded_indices or excluded_paths:
            filtered: list[int] = []
            for idx in indices:
                if idx in excluded_indices:
                    total_excluded_by_index += 1
                    continue
                npath = normalize_path_text(ds.paths[idx])
                if npath in excluded_paths:
                    total_excluded_by_path += 1
                    continue
                filtered.append(idx)
            indices = filtered
        per_class_available_after_exclusions[class_idx] = len(indices)
        if image_values is not None:
            per_class_available_unique_images[class_idx] = len({image_values[i] for i in indices if image_values[i]})
        else:
            per_class_available_unique_images[class_idx] = -1
        if source_values is not None:
            per_class_available_unique_sources[class_idx] = len({source_values[i] for i in indices if source_values[i]})
        else:
            per_class_available_unique_sources[class_idx] = -1
        if args.shuffle_per_class:
            rng = random.Random(args.seed + class_idx)
            rng.shuffle(indices)
        if args.max_per_class is not None:
            if (image_values is not None or source_values is not None) and not args.disable_diversity_selection:
                chosen, uniq_img, uniq_src = select_diverse_indices(
                    indices,
                    args.max_per_class,
                    image_values=image_values,
                    source_values=source_values,
                )
            else:
                chosen = indices[: args.max_per_class]
                if image_values is not None:
                    uniq_img = len({image_values[i] for i in chosen if image_values[i]})
                else:
                    uniq_img = -1
                if source_values is not None:
                    uniq_src = len({source_values[i] for i in chosen if source_values[i]})
                else:
                    uniq_src = -1
        else:
            chosen = indices
            if image_values is not None:
                uniq_img = len({image_values[i] for i in chosen if image_values[i]})
            else:
                uniq_img = -1
            if source_values is not None:
                uniq_src = len({source_values[i] for i in chosen if source_values[i]})
            else:
                uniq_src = -1

        per_class_staged[class_idx] = len(chosen)
        per_class_unique_images[class_idx] = uniq_img
        per_class_unique_sources[class_idx] = uniq_src
        selected_indices.extend(chosen)

    if args.limit_total is not None:
        selected_indices = selected_indices[: args.limit_total]

    print(f"Planned staged rows: {len(selected_indices)}")
    if args.max_per_class is not None:
        print(f"Per-class cap applied: {args.max_per_class}")
        if (image_values is not None or source_values is not None) and not args.disable_diversity_selection:
            print("Per-class selection mode: diversity-aware")
    if excluded_indices or excluded_paths:
        print(
            "Excluded candidates removed before selection: "
            f"by_index={total_excluded_by_index}, by_path={total_excluded_by_path}"
        )

    # Use dataset index in filename so mapping remains stable.
    tasks: list[StageTask] = []
    for dataset_index in selected_indices:
        src_path = ds.paths[dataset_index]
        class_index = ds.target_indices[dataset_index]
        class_label = ds.targets[dataset_index]
        class_folder = class_folder_by_idx[class_index]
        fname = f"{dataset_index:07d}__{Path(src_path).name}"
        dst_path = images_dir / class_folder / fname
        tasks.append(
            StageTask(
                dataset_index=dataset_index,
                class_index=class_index,
                class_label=class_label,
                class_folder=class_folder,
                src_path=src_path,
                dst_path=dst_path,
            )
        )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        iterator = ex.map(
            lambda t: stage_one(
                t,
                check_decode=args.check_image_decode,
                stage_mode=args.stage_mode,
                overwrite=args.overwrite,
            ),
            tasks,
        )
        if not args.no_progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(tasks), desc=f"Staging ({args.workers} workers)")
        for row in iterator:
            rows.append(row)

    result_df = pd.DataFrame(rows)
    ok_df = result_df[result_df["ok"]].copy()
    fail_df = result_df[~result_df["ok"]].copy()
    print(f"Staged ok: {len(ok_df)}")
    print(f"Failed: {len(fail_df)}")
    if len(fail_df) > 0:
        reason_counts = fail_df["reason"].value_counts().to_dict()
        print(f"Failure reasons: {reason_counts}")

    split_col = find_column_case_insensitive(ds._data.columns.tolist(), args.split_column)

    # Save per-sample pool rows with original CSV columns + pool metadata.
    pool_rows: list[dict] = []
    for rec in ok_df.to_dict("records"):
        idx = int(rec["dataset_index"])
        row_dict = ds._data.iloc[idx].to_dict()
        row_dict["_pool_dataset_index"] = idx
        row_dict["_pool_target_index"] = int(rec["class_index"])
        row_dict["_pool_target_label"] = rec["class_label"]
        row_dict["_pool_class_folder"] = rec["class_folder"]
        row_dict["_pool_source_path"] = rec["src_path"]
        row_dict["_pool_staged_path"] = rec["dst_path"]
        if split_col is not None and split_col in ds._data.columns:
            row_dict["_pool_split_value"] = ds._data.iloc[idx][split_col]
        pool_rows.append(row_dict)

    if pool_rows:
        pool_df = pd.DataFrame(pool_rows)
        pool_df.sort_values(by=["_pool_target_index", "_pool_dataset_index"], inplace=True)
    else:
        pool_df = pd.DataFrame(
            columns=[
                "_pool_dataset_index",
                "_pool_target_index",
                "_pool_target_label",
                "_pool_class_folder",
                "_pool_source_path",
                "_pool_staged_path",
            ]
        )
    pool_csv = output_dir / "pool_rows.csv"
    pool_df.to_csv(pool_csv, index=False)

    manifest_csv = output_dir / "staging_manifest.csv"
    result_df.sort_values(by=["class_index", "dataset_index"], inplace=True)
    result_df.to_csv(manifest_csv, index=False)

    errors_csv = output_dir / "staging_errors.csv"
    fail_df.to_csv(errors_csv, index=False)

    # Class summary across filtered source pool and staged subset.
    all_counts = pd.Series(ds.target_indices).value_counts().to_dict()
    staged_counts = ok_df["class_index"].value_counts().to_dict()
    summary_rows = []
    for class_idx, class_name in ds.idx_to_class.items():
        summary_rows.append(
            {
                "class_index": class_idx,
                "class_label": class_name,
                "available_in_filtered_pool": int(all_counts.get(class_idx, 0)),
                "available_after_exclusions": int(per_class_available_after_exclusions.get(class_idx, 0)),
                "available_unique_image_names": int(per_class_available_unique_images.get(class_idx, -1)),
                "available_unique_sources": int(per_class_available_unique_sources.get(class_idx, -1)),
                "planned_for_staging": int(per_class_staged.get(class_idx, 0)),
                "selected_unique_image_names": int(per_class_unique_images.get(class_idx, -1)),
                "selected_unique_sources": int(per_class_unique_sources.get(class_idx, -1)),
                "staged_ok": int(staged_counts.get(class_idx, 0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(by="class_index")
    summary_csv = output_dir / "class_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nOutputs:")
    print(f"- pool rows:      {pool_csv}")
    print(f"- stage manifest: {manifest_csv}")
    print(f"- stage errors:   {errors_csv}")
    print(f"- class summary:  {summary_csv}")
    print(f"- staged images:  {images_dir}")


if __name__ == "__main__":
    main()
