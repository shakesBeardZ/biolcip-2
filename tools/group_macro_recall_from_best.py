#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute grouped macro recall (e.g., head/body/tail) from predictions.csv "
            "pointed to by best-checkpoint CSV rows."
        )
    )
    parser.add_argument(
        "--best-csv",
        required=True,
        help="CSV like best_by_setup_model_macro_recall.csv (must include predictions_csv).",
    )
    parser.add_argument(
        "--groups-json",
        required=True,
        help="JSON mapping group name -> list of class names (genus/family tokens).",
    )
    parser.add_argument(
        "--model",
        default="bioclip",
        help="Model filter in best-csv (default: bioclip).",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        default=["full", "k1", "k5", "k10", "k20"],
        help="Setup filters (default: full k1 k5 k10 k20).",
    )
    parser.add_argument(
        "--output-long-csv",
        required=True,
        help="Output CSV in long format: one row per setup/group.",
    )
    parser.add_argument(
        "--output-wide-csv",
        required=True,
        help="Output CSV in wide format: one row per setup with *_macro_recall columns.",
    )
    return parser.parse_args()


def clean_label(label: str) -> str:
    text = str(label).strip()
    if not text:
        return ""
    # For chain labels like 'Animalia ... Porites', use the leaf token.
    return text.split()[-1].lower()


def per_class_recall(df: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, int]]:
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise KeyError("predictions.csv must contain true_label and pred_label columns.")

    true_leaf = df["true_label"].astype(str).map(clean_label)
    pred_leaf = df["pred_label"].astype(str).map(clean_label)

    recalls: Dict[str, float] = {}
    supports: Dict[str, int] = {}

    for cls in sorted(true_leaf.unique().tolist()):
        true_mask = true_leaf == cls
        support = int(true_mask.sum())
        supports[cls] = support
        if support == 0:
            recalls[cls] = 0.0
            continue
        tp = int((true_mask & (pred_leaf == cls)).sum())
        recalls[cls] = float(tp / support)
    return recalls, supports


def main() -> None:
    args = parse_args()

    best_csv = Path(args.best_csv)
    groups_json = Path(args.groups_json)
    if not best_csv.exists():
        raise FileNotFoundError(f"best-csv not found: {best_csv}")
    if not groups_json.exists():
        raise FileNotFoundError(f"groups-json not found: {groups_json}")

    best_df = pd.read_csv(best_csv, low_memory=False)
    if "predictions_csv" not in best_df.columns:
        raise KeyError("best-csv must contain predictions_csv column.")

    model = str(args.model).strip().lower()
    setups = {s.strip() for s in args.setups if s.strip()}
    filt = best_df[
        best_df["model"].astype(str).str.strip().str.lower().eq(model)
        & best_df["setup"].astype(str).str.strip().isin(setups)
    ].copy()
    if filt.empty:
        raise RuntimeError(f"No rows found for model={model}, setups={sorted(setups)}")

    with groups_json.open("r", encoding="utf-8") as f:
        groups_raw = json.load(f)
    if not isinstance(groups_raw, dict) or not groups_raw:
        raise ValueError("groups-json must be a non-empty object: {group: [labels...]}")

    group_map: Dict[str, List[str]] = {
        g: [str(x).strip().lower() for x in labels if str(x).strip()]
        for g, labels in groups_raw.items()
    }

    long_rows = []
    wide_rows = []

    # Stable ordering by setup priority.
    setup_order = {k: i for i, k in enumerate(["full", "k1", "k5", "k10", "k20", "zero_shot"])}
    filt = filt.sort_values(
        by=["setup", "model"],
        key=lambda s: s.map(lambda x: setup_order.get(str(x), 999)),
    )

    for _, row in filt.iterrows():
        setup = str(row.get("setup", ""))
        run_name = str(row.get("run_name", ""))
        ckpt_name = str(row.get("checkpoint_name", ""))
        ckpt_path = str(row.get("checkpoint_path", ""))
        pred_csv = Path(str(row.get("predictions_csv", "")))
        if not pred_csv.exists():
            raise FileNotFoundError(f"predictions.csv not found: {pred_csv}")

        pred_df = pd.read_csv(pred_csv, low_memory=False)
        recalls, supports = per_class_recall(pred_df)

        wide = {
            "setup": setup,
            "model": model,
            "run_name": run_name,
            "checkpoint_name": ckpt_name,
            "checkpoint_path": ckpt_path,
            "predictions_csv": str(pred_csv),
        }

        for group_name, classes in group_map.items():
            present_classes = [c for c in classes if supports.get(c, 0) > 0]
            total_support = int(sum(supports.get(c, 0) for c in classes))
            if present_classes:
                grp_macro = float(sum(recalls[c] for c in present_classes) / len(present_classes))
            else:
                grp_macro = float("nan")

            long_rows.append(
                {
                    "setup": setup,
                    "model": model,
                    "run_name": run_name,
                    "checkpoint_name": ckpt_name,
                    "checkpoint_path": ckpt_path,
                    "group": group_name,
                    "group_macro_recall": grp_macro,
                    "present_classes": len(present_classes),
                    "group_classes_total": len(classes),
                    "group_support": total_support,
                    "predictions_csv": str(pred_csv),
                }
            )

            wide[f"{group_name}_macro_recall"] = grp_macro
            wide[f"{group_name}_present_classes"] = len(present_classes)
            wide[f"{group_name}_group_classes_total"] = len(classes)
            wide[f"{group_name}_support"] = total_support

        wide_rows.append(wide)

    out_long = Path(args.output_long_csv)
    out_wide = Path(args.output_wide_csv)
    out_long.parent.mkdir(parents=True, exist_ok=True)
    out_wide.parent.mkdir(parents=True, exist_ok=True)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    long_df.to_csv(out_long, index=False)
    wide_df.to_csv(out_wide, index=False)

    print(f"Wrote: {out_long}")
    print(f"Wrote: {out_wide}")
    print(f"Rows (long): {len(long_df)}")
    print(f"Rows (wide): {len(wide_df)}")


if __name__ == "__main__":
    main()
