#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check missing/unreadable image files from a coral CSV in parallel."
    )
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument(
        "--path-key",
        default=None,
        help="Column name for image paths (case-insensitive). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--split-column",
        default=None,
        help="Optional split column to filter rows (e.g., split).",
    )
    parser.add_argument(
        "--include-splits",
        default=None,
        help="Comma-separated split values to keep (e.g., train,val).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional root dir prepended to relative paths.",
    )
    parser.add_argument(
        "--path-replace-from",
        default=None,
        help="Optional string/prefix to replace in each resolved path.",
    )
    parser.add_argument(
        "--path-replace-to",
        default=None,
        help="Replacement for --path-replace-from.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Parallel workers for path checks (default: 24).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to check.",
    )
    parser.add_argument(
        "--check-readable",
        action="store_true",
        help="Also check file readability by reading one byte.",
    )
    parser.add_argument(
        "--check-image-decode",
        action="store_true",
        help="Also verify image decode with PIL.Image.verify().",
    )
    parser.add_argument(
        "--output-missing-csv",
        default="missing_files.csv",
        help="Output CSV containing only missing/failed rows.",
    )
    parser.add_argument(
        "--output-checked-csv",
        default=None,
        help="Optional output CSV containing all checked rows with status columns.",
    )
    parser.add_argument(
        "--show-first",
        type=int,
        default=20,
        help="How many missing rows to print in the console (default: 20).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output.",
    )
    return parser.parse_args()


def resolve_column_name(columns: list[str], requested: Optional[str]) -> str:
    cols_lower = {c.lower(): c for c in columns}
    candidates: list[str] = []
    if requested:
        candidates.append(requested)
    candidates.extend(["local_path", "filepath", "path", "image_path", "image_file", "patch_path"])
    for candidate in candidates:
        found = cols_lower.get(candidate.lower())
        if found:
            return found
    raise KeyError(
        "Could not find path column. Provide --path-key or ensure one of: "
        "local_path, filepath, path, image_path, image_file, patch_path."
    )


def apply_split_filter(df: pd.DataFrame, split_column: Optional[str], include_splits: Optional[str]) -> pd.DataFrame:
    if not split_column:
        return df

    split_col = None
    target = split_column.lower()
    for col in df.columns:
        if col.lower() == target:
            split_col = col
            break
    if split_col is None:
        raise KeyError(f"Split column '{split_column}' not found in CSV.")

    if not include_splits:
        raise ValueError("When --split-column is provided, --include-splits must also be provided.")

    values = {s.strip().lower() for s in include_splits.split(",") if s.strip()}
    if not values:
        raise ValueError("No valid split values parsed from --include-splits.")

    filtered = df[df[split_col].astype(str).str.strip().str.lower().isin(values)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def resolve_path(
    raw_value: object,
    data_root: Optional[str],
    replace_from: Optional[str],
    replace_to: Optional[str],
) -> str:
    if pd.isna(raw_value):
        path = ""
    else:
        path = str(raw_value).strip()

    if path and data_root and not os.path.isabs(path):
        path = os.path.join(data_root, path)

    if replace_from is not None:
        path = path.replace(replace_from, replace_to or "")

    return path


def _check_one(item: tuple[str, bool, bool]) -> tuple[bool, str]:
    path, check_readable, check_image_decode = item
    if not path:
        return False, "empty_path"
    if not os.path.isfile(path):
        return False, "missing"

    try:
        if check_image_decode:
            from PIL import Image

            with Image.open(path) as img:
                # Match dataset_train.py behavior: force full decode via convert().
                _ = img.convert("RGB")
        elif check_readable:
            with open(path, "rb") as f:
                _ = f.read(1)
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"

    return True, "ok"


def main() -> None:
    args = parse_args()

    if (args.path_replace_from is None) != (args.path_replace_to is None):
        raise ValueError("--path-replace-from and --path-replace-to must be provided together.")
    if args.path_replace_from == "":
        raise ValueError("--path-replace-from cannot be empty.")
    if args.check_image_decode and not args.check_readable:
        # decode implies readable
        args.check_readable = True

    df = pd.read_csv(args.csv, low_memory=False)
    if df.empty:
        raise ValueError(f"CSV has no rows: {args.csv}")

    df["_source_row_index"] = df.index
    df = apply_split_filter(df, args.split_column, args.include_splits)
    if df.empty:
        raise ValueError("No rows left after applying filters.")

    path_column = resolve_column_name(df.columns.tolist(), args.path_key)
    df["_resolved_path"] = [
        resolve_path(v, args.data_root, args.path_replace_from, args.path_replace_to)
        for v in df[path_column].tolist()
    ]

    if args.limit is not None:
        df = df.head(args.limit).copy()

    items = [
        (p, args.check_readable, args.check_image_decode) for p in df["_resolved_path"].tolist()
    ]

    total = len(items)
    exists_list: list[bool] = []
    reason_list: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        iterator = ex.map(_check_one, items)
        if not args.no_progress and tqdm is not None:
            iterator = tqdm(iterator, total=total, desc=f"Checking ({args.workers} workers)")
        for exists, reason in iterator:
            exists_list.append(exists)
            reason_list.append(reason)

    df["_exists"] = exists_list
    df["_check_reason"] = reason_list

    missing_df = df[~df["_exists"]].copy()
    missing_count = len(missing_df)
    print(f"Checked rows: {total}")
    print(f"Missing/failed rows: {missing_count}")
    print(f"Present rows: {total - missing_count}")

    if missing_count:
        preview_cols = ["_source_row_index", path_column, "_resolved_path", "_check_reason"]
        print("\nFirst missing rows:")
        print(missing_df[preview_cols].head(args.show_first).to_string(index=False))

    if args.output_missing_csv:
        out_missing = Path(args.output_missing_csv)
        out_missing.parent.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(out_missing, index=False)
        print(f"\nWrote missing rows CSV: {out_missing}")

    if args.output_checked_csv:
        out_checked = Path(args.output_checked_csv)
        out_checked.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_checked, index=False)
        print(f"Wrote full checked CSV: {out_checked}")


if __name__ == "__main__":
    main()
