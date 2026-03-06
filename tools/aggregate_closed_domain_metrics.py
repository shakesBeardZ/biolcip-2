#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class EvalMeta:
    setup: str
    model: str
    run_name: str
    checkpoint_name: str
    epoch: Optional[int]
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate closed-domain evaluation outputs into a metrics CSV and best-checkpoint tables. "
            "Computes acc1 and macro recall from predictions.csv."
        )
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Root folder containing eval outputs (e.g., /raid/.../closed_domain_checkpoint_matrix).",
    )
    parser.add_argument(
        "--predictions-name",
        default="predictions.csv",
        help="Prediction filename to scan recursively (default: predictions.csv).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV for all metrics rows. Default: <base-dir>/metrics_table.csv",
    )
    parser.add_argument(
        "--best-acc1-csv",
        default=None,
        help="Output CSV of best checkpoint per (setup, model) by acc1. "
             "Default: <base-dir>/best_by_setup_model_acc1.csv",
    )
    parser.add_argument(
        "--best-macro-recall-csv",
        default=None,
        help="Output CSV of best checkpoint per (setup, model) by macro recall. "
             "Default: <base-dir>/best_by_setup_model_macro_recall.csv",
    )
    parser.add_argument(
        "--write-summary-json",
        action="store_true",
        help="Write per-eval summary JSON (with per_class stats) next to each predictions.csv.",
    )
    parser.add_argument(
        "--summary-json-name",
        default="new_summary.json",
        help="Name of summary JSON when --write-summary-json is enabled (default: new_summary.json).",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        default=None,
        help="Optional setup filter (e.g., full k1 k5 k10 k20 zero_shot).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model filter (e.g., bioclip clip openclip).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first bad predictions file. Default is to skip bad files with a warning.",
    )
    return parser.parse_args()


def discover_prediction_files(base_dir: Path, predictions_name: str) -> list[Path]:
    """
    Find prediction CSVs under base_dir.

    Primary scan uses Path.rglob (fast and simple), but it does not recurse into
    symlinked directories. If nothing is found, fall back to os.walk with
    followlinks=True so merged views built from symlinks are supported.
    """
    pred_files = sorted(base_dir.rglob(predictions_name))
    if pred_files:
        return pred_files

    files: list[Path] = []
    for root, _, filenames in os.walk(base_dir, followlinks=True):
        if predictions_name in filenames:
            files.append(Path(root) / predictions_name)
    return sorted(files)


def infer_meta(base_dir: Path, predictions_path: Path) -> EvalMeta:
    rel = predictions_path.relative_to(base_dir)
    parts = rel.parts

    setup = "unknown"
    model = "unknown"
    run_name = ""
    checkpoint_name = ""
    checkpoint_path = ""
    epoch: Optional[int] = None

    # Pattern A:
    #   <setup>/<model>/<run_name>/<checkpoint_name>/predictions.csv
    # Pattern B:
    #   zero_shot/<model>/<run_name>/predictions.csv
    if len(parts) >= 5 and (parts[0] == "full" or re.fullmatch(r"k\d+", parts[0])):
        setup = parts[0]
        model = parts[1]
        run_name = parts[2]
        checkpoint_name = parts[3]
    elif len(parts) >= 4 and parts[0] == "zero_shot":
        setup = "zero_shot"
        model = parts[1]
        run_name = parts[2]
        checkpoint_name = "pretrained"
    elif len(parts) >= 4:
        # Generic fallback for similarly structured outputs
        setup = parts[0]
        model = parts[1]
        run_name = parts[2]
        checkpoint_name = parts[3]
    elif len(parts) >= 3:
        setup = parts[0]
        model = parts[1]
        run_name = parts[2]
        checkpoint_name = "pretrained"

    # Refine from metrics.json if available.
    metrics_json = predictions_path.with_name("metrics.json")
    if metrics_json.exists():
        try:
            m = json.loads(metrics_json.read_text())
            ckpt = str(m.get("checkpoint", "") or "")
            if ckpt:
                checkpoint_path = ckpt
                checkpoint_name = Path(ckpt).name
                if checkpoint_name.endswith(".pt"):
                    checkpoint_name = checkpoint_name[:-3]
        except Exception:
            pass

    # Derive epoch number from checkpoint name (epoch_12 -> 12).
    match = re.search(r"epoch_(\d+)", checkpoint_name)
    if match:
        epoch = int(match.group(1))
    else:
        epoch = None

    return EvalMeta(
        setup=setup,
        model=model,
        run_name=run_name,
        checkpoint_name=checkpoint_name or "pretrained",
        epoch=epoch,
        checkpoint_path=checkpoint_path,
    )


def compute_metrics(df: pd.DataFrame) -> tuple[float, float, int, dict]:
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise KeyError("predictions.csv must contain columns: true_label, pred_label")

    true = df["true_label"].astype(str)
    pred = df["pred_label"].astype(str)
    n = int(len(df))
    if n == 0:
        raise ValueError("predictions.csv has 0 rows")

    if "correct_top1" in df.columns:
        acc1 = float(pd.to_numeric(df["correct_top1"], errors="coerce").fillna(0).mean())
    else:
        acc1 = float((true == pred).mean())

    classes = sorted(true.unique().tolist())
    per_class: dict[str, dict] = {}
    recalls: list[float] = []
    for cls in classes:
        true_mask = (true == cls)
        pred_mask = (pred == cls)
        support = int(true_mask.sum())
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / support) if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        # Matches your prior method: macro over classes present in per_class.
        recalls.append(float(recall))

    macro_recall = float(sum(recalls) / len(recalls))
    return acc1, macro_recall, n, per_class


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"base-dir does not exist: {base_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else base_dir / "metrics_table.csv"
    best_acc1_csv = Path(args.best_acc1_csv) if args.best_acc1_csv else base_dir / "best_by_setup_model_acc1.csv"
    best_macro_csv = (
        Path(args.best_macro_recall_csv)
        if args.best_macro_recall_csv
        else base_dir / "best_by_setup_model_macro_recall.csv"
    )

    pred_files = discover_prediction_files(base_dir, args.predictions_name)
    if not pred_files:
        raise RuntimeError(f"No {args.predictions_name} files found under {base_dir}")

    rows: list[dict] = []
    skipped = 0
    for pred_path in pred_files:
        try:
            meta = infer_meta(base_dir, pred_path)
            if args.setups and meta.setup not in set(args.setups):
                continue
            if args.models and meta.model not in set(args.models):
                continue

            df = pd.read_csv(pred_path, low_memory=False)
            acc1, macro_recall, support, per_class = compute_metrics(df)

            row = {
                "model": meta.model,
                "setup": meta.setup,
                "run_name": meta.run_name,
                "checkpoint_name": meta.checkpoint_name,
                "epoch": meta.epoch if meta.epoch is not None else "",
                "macro_recall": macro_recall,
                "acc1": acc1,
                "support": support,
                "num_classes": len(per_class),
                "predictions_csv": str(pred_path),
                "checkpoint_path": meta.checkpoint_path,
            }
            rows.append(row)

            if args.write_summary_json:
                summary = {
                    "setup": meta.setup,
                    "model": meta.model,
                    "run_name": meta.run_name,
                    "checkpoint_name": meta.checkpoint_name,
                    "checkpoint_path": meta.checkpoint_path,
                    "top1_accuracy": acc1,
                    "macro_recall": macro_recall,
                    "support": support,
                    "num_classes": len(per_class),
                    "per_class": per_class,
                }
                out_json = pred_path.with_name(args.summary_json_name)
                out_json.write_text(json.dumps(summary, indent=2))
        except Exception as exc:
            msg = f"[WARN] Skipping {pred_path}: {type(exc).__name__}: {exc}"
            if args.strict:
                raise RuntimeError(msg) from exc
            print(msg)
            skipped += 1

    if not rows:
        raise RuntimeError("No valid prediction files were aggregated.")

    all_df = pd.DataFrame(rows)
    # Stable ordering for readability.
    all_df["epoch_sort"] = pd.to_numeric(all_df["epoch"], errors="coerce").fillna(-1)
    all_df = all_df.sort_values(
        by=["setup", "model", "run_name", "epoch_sort", "checkpoint_name"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["epoch_sort"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_csv, index=False)

    # Best by acc1 per (setup, model).
    best_acc1 = (
        all_df.sort_values(by=["setup", "model", "acc1", "macro_recall"], ascending=[True, True, False, False])
        .groupby(["setup", "model"], as_index=False)
        .first()
    )
    best_acc1.to_csv(best_acc1_csv, index=False)

    # Best by macro recall per (setup, model).
    best_macro = (
        all_df.sort_values(by=["setup", "model", "macro_recall", "acc1"], ascending=[True, True, False, False])
        .groupby(["setup", "model"], as_index=False)
        .first()
    )
    best_macro.to_csv(best_macro_csv, index=False)

    print(f"Aggregated files: {len(all_df)}")
    print(f"Skipped files: {skipped}")
    print(f"Wrote: {output_csv}")
    print(f"Wrote: {best_acc1_csv}")
    print(f"Wrote: {best_macro_csv}")


if __name__ == "__main__":
    main()
