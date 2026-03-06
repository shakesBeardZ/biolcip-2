#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic few-shot training CSVs (k=1/5/10) from staged manual pool, "
            "and export a fixed test CSV for closed-domain zero-shot/eval."
        )
    )
    parser.add_argument("--source-csv", required=True, help="Original full coral CSV.")
    parser.add_argument(
        "--pool-csv",
        required=True,
        help="Pool rows CSV from tools/stage_manual_fewshot_pool.py (typically data/fewshot_manual_pool/pool_rows.csv).",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for generated CSVs.")
    parser.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Few-shot K values to export.",
    )
    parser.add_argument(
        "--class-column",
        default="_pool_target_label",
        help="Column in pool CSV that defines class grouping.",
    )
    parser.add_argument(
        "--order-column",
        default="_pool_dataset_index",
        help="Column used to deterministically rank examples inside each class.",
    )
    parser.add_argument(
        "--train-split-column",
        default="split",
        help="Split column name written into few-shot train CSVs.",
    )
    parser.add_argument(
        "--train-split-value",
        default="train",
        help="Split value for rows in few-shot train CSVs.",
    )
    parser.add_argument(
        "--source-split-column",
        default="Split",
        help="Split column name in source CSV used to extract fixed test rows.",
    )
    parser.add_argument(
        "--test-split-value",
        default="test",
        help="Split value in source CSV to keep for fixed test export.",
    )
    parser.add_argument(
        "--path-source-column",
        default="patch_path",
        help="Path column in CSVs used to populate Path when needed.",
    )
    parser.add_argument(
        "--path-out-column",
        default="Path",
        help="Output path column for evaluation CSVs.",
    )
    parser.add_argument("--path-replace-from", default=None, help="Optional path replace source.")
    parser.add_argument("--path-replace-to", default=None, help="Optional path replace target.")
    parser.add_argument(
        "--drop-pool-helper-columns",
        action="store_true",
        help="Drop _pool_* helper columns from exported few-shot train CSVs.",
    )
    return parser.parse_args()


def ci_column(columns: list[str], target: str) -> Optional[str]:
    t = target.lower()
    for c in columns:
        if c.lower() == t:
            return c
    return None


def normalize_path_series(
    series: pd.Series,
    replace_from: Optional[str],
    replace_to: Optional[str],
) -> pd.Series:
    out = series.astype(str).str.strip()
    if replace_from is not None:
        out = out.str.replace(replace_from, replace_to or "", regex=False)
    return out


def main() -> None:
    args = parse_args()
    if args.path_replace_from is not None and args.path_replace_to is None:
        raise ValueError("--path-replace-to is required when --path-replace-from is set.")
    if args.path_replace_from is None and args.path_replace_to is not None:
        raise ValueError("--path-replace-from is required when --path-replace-to is set.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = sorted(set(args.k_list))
    if any(k <= 0 for k in ks):
        raise ValueError("All k values in --k-list must be positive.")
    kmax = max(ks)

    pool_df = pd.read_csv(args.pool_csv, low_memory=False)
    if pool_df.empty:
        raise ValueError(f"Pool CSV has no rows: {args.pool_csv}")

    class_col = ci_column(pool_df.columns.tolist(), args.class_column)
    if class_col is None:
        raise KeyError(f"Class column not found in pool CSV: {args.class_column}")

    order_col = ci_column(pool_df.columns.tolist(), args.order_column)
    if order_col is None:
        raise KeyError(f"Order column not found in pool CSV: {args.order_column}")

    pool_df = pool_df.sort_values(by=[class_col, order_col]).reset_index(drop=True)
    pool_df["_fewshot_rank"] = pool_df.groupby(class_col).cumcount() + 1

    counts = pool_df[class_col].value_counts()
    too_small = counts[counts < kmax]
    if not too_small.empty:
        details = ", ".join(f"{cls}={int(n)}" for cls, n in too_small.items())
        raise ValueError(f"Pool does not have enough rows for k={kmax}: {details}")

    # Build few-shot train CSVs.
    pool_path_src_col = ci_column(pool_df.columns.tolist(), args.path_source_column)
    if pool_path_src_col is None:
        raise KeyError(f"Path source column not found in pool CSV: {args.path_source_column}")

    for k in ks:
        k_df = pool_df[pool_df["_fewshot_rank"] <= k].copy()
        k_df[args.train_split_column] = args.train_split_value
        if "Split" in k_df.columns:
            k_df["Split"] = args.train_split_value

        # Ensure evaluation-friendly path column exists too.
        k_df[args.path_out_column] = normalize_path_series(
            k_df[pool_path_src_col], args.path_replace_from, args.path_replace_to
        )

        if args.drop_pool_helper_columns:
            drop_cols = [c for c in k_df.columns if c.startswith("_pool_")]
            k_df = k_df.drop(columns=drop_cols)

        out_csv = out_dir / f"fewshot_train_k{k}.csv"
        k_df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} (rows={len(k_df)}, classes={k_df[class_col].nunique()})")

    # Export fixed test CSV from source data for closed-domain evaluation.
    src_df = pd.read_csv(args.source_csv, low_memory=False)
    split_col = ci_column(src_df.columns.tolist(), args.source_split_column)
    if split_col is None:
        raise KeyError(f"Source split column not found: {args.source_split_column}")

    test_mask = src_df[split_col].astype(str).str.strip().str.lower().eq(args.test_split_value.lower())
    test_df = src_df[test_mask].copy()
    if test_df.empty:
        raise ValueError(f"No rows found in source CSV where {split_col} == {args.test_split_value}")

    src_path_col = ci_column(test_df.columns.tolist(), args.path_source_column)
    if src_path_col is None:
        raise KeyError(f"Path source column not found in source CSV: {args.path_source_column}")

    test_df[args.path_out_column] = normalize_path_series(
        test_df[src_path_col], args.path_replace_from, args.path_replace_to
    )
    test_out = out_dir / "test_fixed_catalogue.csv"
    test_df.to_csv(test_out, index=False)
    print(f"Wrote {test_out} (rows={len(test_df)})")

    # Helpful summary table.
    summary_rows = []
    for k in ks:
        k_csv = out_dir / f"fewshot_train_k{k}.csv"
        k_df = pd.read_csv(k_csv, low_memory=False)
        summary_rows.append(
            {
                "split_name": f"fewshot_train_k{k}",
                "rows": len(k_df),
                "classes": int(k_df[class_col].nunique()) if class_col in k_df.columns else -1,
            }
        )
    summary_rows.append({"split_name": "test_fixed_catalogue", "rows": len(test_df), "classes": -1})
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "fewshot_split_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
