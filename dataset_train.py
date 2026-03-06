from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.imageomics.naming_eval import Taxon


class CoralTrainingDataset(Dataset):
    """Species-focused dataset loader for BioCLIP-2 fine-tuning.

    The loader expects a CSV file with at least the following taxonomy columns:
    ``kingdom, phylum, class, order, family, genus, species`` plus one column
    containing the local image path (defaults to ``local_path``).

    Additional descriptive columns (e.g. ``characters``, ``colour``) are preserved
    and returned as metadata for downstream prompt engineering.

    Parameters
    ----------
    csv_file:
        Path to the annotations CSV.
    data_root:
        Optional directory that is prepended when resolving relative image paths.
    transform:
        Optional torchvision-style transform applied to the PIL image (RGB).
    target_level:
        Taxonomic rank used to build class indices (default: ``"species"``).
    rank_only:
        If ``True``, the target label is only the chosen rank (e.g. ``"Acropora"``
        for genus). If ``False``, the label contains the taxonomic chain up to the
        target rank.
    caption_level:
        Taxonomic rank used to generate the training caption string. Defaults to
        the same value as ``target_level``.
    caption_mode:
        Either ``"chain"`` (taxonomy chain up to ``caption_level``) or
        ``"rank_only"`` (just the selected rank; species -> scientific name).
    caption_prefix:
        Optional string prepended to each caption (useful for domain prompts).
    normalize_anthozoa:
        Replace ``class=Hexacorallia`` with ``Anthozoa`` when ``phylum=Cnidaria``.
    metadata_columns:
        Optional subset of CSV columns to keep as metadata. When ``None``, all
        non-taxonomy columns (excluding the image path column) are retained.
    path_key:
        Column name (case-insensitive) pointing to the local image path. Defaults
        to ``local_path`` with fallbacks to ``filepath`` or ``path``.
    return_metadata:
        When ``True`` (default) include per-sample metadata in ``__getitem__``.
    drop_missing_targets:
        Drop rows that do not produce a valid target label. When ``False``,
        rows missing the requested rank are kept but receive ``target_idx = -1``.
    infer_genus_from_species:
        If ``True`` (default), infer missing genus from the first token of a
        binomial species name (e.g. ``"Acropora humilis"`` -> genus ``"Acropora"``).
        Set to ``False`` to keep genus empty when genus is missing in CSV.
    path_replace_from / path_replace_to:
        Optional string pair applied to each resolved path via ``str.replace``.
        Useful to remap storage roots across environments.
    split_column:
        Optional column used to filter the CSV (e.g. ``split``).
    include_splits:
        Values from ``split_column`` to keep (case-insensitive). Required when
        ``split_column`` is provided.
    """

    LEVELS_ORDER: Tuple[str, ...] = ("kingdom", "phylum", "cls", "order", "family", "genus", "species")

    def __init__(
        self,
        csv_file: str,
        data_root: Optional[str] = None,
        transform=None,
        target_level: str = "species",
        rank_only: bool = True,
        caption_level: Optional[str] = None,
        caption_mode: str = "chain",
        caption_prefix: Optional[str] = None,
        normalize_anthozoa: bool = True,
        metadata_columns: Optional[Sequence[str]] = None,
        path_key: Optional[str] = None,
        return_metadata: bool = True,
        drop_missing_targets: bool = True,
        split_column: Optional[str] = None,
        include_splits: Optional[Sequence[str]] = None,
        infer_genus_from_species: bool = True,
        path_replace_from: Optional[str] = None,
        path_replace_to: Optional[str] = None,
    ) -> None:
        self.csv_file = csv_file
        self.data_root = data_root
        self.transform = transform
        self.target_level = target_level.lower()
        self.rank_only = rank_only
        self.caption_level = (caption_level or target_level).lower()
        self.caption_mode = caption_mode
        self.caption_prefix = caption_prefix.strip() if caption_prefix else None
        self.normalize_anthozoa = normalize_anthozoa
        self.return_metadata = return_metadata
        self.infer_genus_from_species = infer_genus_from_species
        self.path_replace_from = path_replace_from
        self.path_replace_to = path_replace_to
        if (self.path_replace_from is None) != (self.path_replace_to is None):
            raise ValueError("path_replace_from and path_replace_to must be provided together.")
        if self.path_replace_from == "":
            raise ValueError("path_replace_from cannot be an empty string.")

        self._data = pd.read_csv(csv_file)
        if self._data.empty:
            raise ValueError(f"CSV has no rows: {csv_file}")

        # Optional split filtering before any taxonomy processing
        if split_column is not None:
            split_col = None
            target_lower = split_column.lower()
            for col in self._data.columns:
                if col.lower() == target_lower:
                    split_col = col
                    break
            if split_col is None:
                raise KeyError(f"Split column '{split_column}' not found in CSV.")

            if include_splits is None:
                raise ValueError("When specifying split_column, include_splits must be provided.")

            values = {str(v).strip().lower() for v in include_splits}
            mask = self._data[split_col].astype(str).str.strip().str.lower().isin(values)
            self._data = self._data[mask].reset_index(drop=True)
            if self._data.empty:
                raise ValueError(
                    f"No rows remain after filtering '{split_column}' for values: {sorted(values)}"
                )

        # Normalise column names; keep original for metadata extraction.
        cols_lower = {c.lower(): c for c in self._data.columns}

        # Handle taxonomy column renaming (class -> cls, etc.).
        if "class" in cols_lower and "cls" not in cols_lower:
            self._data.rename(columns={cols_lower["class"]: "cls"}, inplace=True)
            cols_lower = {c.lower(): c for c in self._data.columns}

        for col in self.LEVELS_ORDER:
            if col in cols_lower:
                src = cols_lower[col]
                if src != col:
                    self._data.rename(columns={src: col}, inplace=True)
            else:
                # Create missing taxonomy columns filled with empty strings
                self._data[col] = ""
        cols_lower = {c.lower(): c for c in self._data.columns}

        # Resolve image path column with sensible fallbacks.
        path_candidates = []
        if path_key is not None:
            path_candidates.append(path_key)
        path_candidates.extend(["local_path", "filepath", "path", "image_path", "image_file"])
        resolved_path_col = None
        for candidate in path_candidates:
            candidate_lower = candidate.lower()
            if candidate_lower in cols_lower:
                resolved_path_col = cols_lower[candidate_lower]
                break
        if resolved_path_col is None:
            raise KeyError(
                "Unable to find image path column. Specify `path_key` or add one"
                " of: local_path, filepath, path, image_path, image_file."
            )
        self.path_column = resolved_path_col

        # Prepare metadata (before filtering) by dropping taxonomy + path columns.
        taxonomy_cols = set(self.LEVELS_ORDER) | {self.path_column, "image_url"}
        if metadata_columns is not None:
            self.metadata_columns = [c for c in metadata_columns if c in self._data.columns]
        else:
            self.metadata_columns = [
                c for c in self._data.columns if c not in taxonomy_cols
            ]

        # Build Taxon objects per row.
        self._taxa: List[Taxon] = []
        for _, row in self._data.iterrows():
            taxon = self._row_to_taxon(row)
            self._taxa.append(taxon)

        # Generate targets and captions.
        targets: List[str] = []
        captions: List[str] = []
        keep_indices: List[int] = []

        target_levels: List[str] = []
        caption_levels: List[str] = []

        for idx, taxon in enumerate(self._taxa):
            effective_target_level = self.target_level
            target_label = self._label_from_taxon(taxon, effective_target_level, self.rank_only)
            if not target_label:
                if drop_missing_targets:
                    continue
                fallback_level = self._find_fallback_level(taxon, effective_target_level)
                if fallback_level is None:
                    continue
                effective_target_level = fallback_level
                target_label = self._label_from_taxon(taxon, effective_target_level, self.rank_only)

            effective_caption_level = self.caption_level
            fallback_caption_level = self._find_fallback_level(taxon, effective_caption_level)
            if fallback_caption_level is not None:
                effective_caption_level = fallback_caption_level
            caption_text = self._caption_from_taxon(taxon, level_override=effective_caption_level)
            if self.caption_prefix:
                caption_text = f"{self.caption_prefix} {caption_text}".strip()

            targets.append(target_label)
            captions.append(caption_text)
            target_levels.append(effective_target_level)
            caption_levels.append(effective_caption_level)
            keep_indices.append(idx)

        if len(targets) == 0:
            raise RuntimeError("No valid samples after processing targets.")

        # Filter data/taxa/metadata according to kept indices.
        if len(keep_indices) != len(self._data):
            self._data = self._data.iloc[keep_indices].reset_index(drop=True)
            self._taxa = [self._taxa[i] for i in keep_indices]

        # Resolve absolute paths.
        self.paths: List[str] = []
        for p in self._data[self.path_column].astype(str).tolist():
            path = p.strip()
            if self.data_root is not None and not os.path.isabs(path):
                path = os.path.join(self.data_root, path)
            if self.path_replace_from is not None:
                path = path.replace(self.path_replace_from, self.path_replace_to)
            self.paths.append(path)

        self.captions = captions
        self.targets = targets

        unique_classes = sorted({t for t in targets if t})
        self.class_to_idx: Dict[str, int] = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class: Dict[int, str] = {i: cls for cls, i in self.class_to_idx.items()}
        self.classes = unique_classes
        self.allow_missing_targets = not drop_missing_targets
        self.target_indices = []
        for t in targets:
            if t:
                self.target_indices.append(self.class_to_idx[t])
            else:
                self.target_indices.append(-1)
        if any(i < 0 for i in self.target_indices) and drop_missing_targets:
            raise RuntimeError("Found sample without valid class index; check target generation.")

        self.samples: List[Tuple[str, int]] = list(zip(self.paths, self.target_indices))

        if self.return_metadata and self.metadata_columns:
            self._metadata: List[Dict[str, Any]] = []
            for _, row in self._data.iterrows():
                meta = {col: row[col] for col in self.metadata_columns if col in row}
                self._metadata.append(meta)
        else:
            self._metadata = []

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _row_to_taxon(self, row: pd.Series) -> Taxon:
        def _get(col: str) -> str:
            val = row.get(col, "")
            return str(val).strip() if pd.notna(val) else ""

        kingdom = _get("kingdom")
        phylum = _get("phylum")
        cls_val = _get("cls") or _get("class")
        order = _get("order")
        family = _get("family")
        genus = _get("genus")
        species_raw = _get("species")

        # Extract species epithet (avoid duplicating genus).
        species_ep = ""
        if species_raw:
            tokens = species_raw.split()
            if len(tokens) >= 2:
                if self.infer_genus_from_species and not genus:
                    genus = tokens[0]
                if genus and tokens[0].lower() == genus.lower():
                    species_ep = tokens[1]
                else:
                    species_ep = tokens[-1]
            else:
                species_ep = tokens[0]

        if self.normalize_anthozoa:
            if phylum.lower() == "cnidaria" and cls_val.lower() == "hexacorallia":
                cls_val = "Anthozoa"

        return Taxon(
            kingdom=kingdom,
            phylum=phylum,
            cls=cls_val,
            order=order,
            family=family,
            genus=genus,
            species=species_ep,
        )

    def _label_from_taxon(self, taxon: Taxon, level: str, rank_only: bool) -> str:
        level = level.lower()
        if level == "species":
            return taxon.scientific_name or ""
        if level == "scientific":
            return taxon.scientific_name or ""
        if level not in self.LEVELS_ORDER:
            raise ValueError(f"Unsupported target level: {level}")

        tpl = taxon.to_tuple()
        idx = self.LEVELS_ORDER.index(level)
        if rank_only:
            return tpl[idx] if idx < len(tpl) else ""
        parts = [p for i, p in enumerate(tpl) if i <= idx and p]
        return " ".join(parts).strip()

    def _find_fallback_level(self, taxon: Taxon, level: str) -> Optional[str]:
        tpl = taxon.to_tuple()
        idx = self.LEVELS_ORDER.index(level)
        for i in range(min(idx, len(tpl) - 1), -1, -1):
            if tpl[i]:
                return self.LEVELS_ORDER[i]
        return None

    def _caption_from_taxon(self, taxon: Taxon, level_override: Optional[str] = None) -> str:
        level = level_override or self.caption_level
        mode = self.caption_mode.lower()
        if level == "species" and mode == "rank_only":
            return taxon.scientific_name or ""
        if level == "scientific":
            return taxon.scientific_name or ""
        if level not in self.LEVELS_ORDER:
            raise ValueError(f"Unsupported caption level: {level}")

        tpl = taxon.to_tuple()
        idx = self.LEVELS_ORDER.index(level)
        if mode == "rank_only":
            if idx == len(self.LEVELS_ORDER) - 1:
                return taxon.scientific_name or ""
            return tpl[idx] if idx < len(tpl) else ""
        if mode == "chain":
            parts = [p for i, p in enumerate(tpl) if i <= idx and p]
            return " ".join(parts).strip()
        raise ValueError(f"Unsupported caption_mode: {self.caption_mode}")

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        caption = self.captions[index]
        target_idx = self.target_indices[index]
        taxon = self._taxa[index]
        # print(f"Loading image at index {index}: {path}")
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image at index {index}: {path}") from exc
            
        if self.transform is not None:
            img = self.transform(img)

        if isinstance(img, torch.Tensor):
            if torch.isnan(img).any() or torch.isinf(img).any():
                raise ValueError(f"Invalid tensor for image at index {index}: {path}")

        sample: Dict[str, Any] = {
            "image": img,
            "text": caption,
            "target": target_idx,
            "target_name": self.targets[index],
            "path": path,
            "taxon": asdict(taxon),
        }
        if self.return_metadata and self._metadata:
            sample["metadata"] = self._metadata[index]
        return sample

    def __repr__(self) -> str:
        return (
            f"CoralTrainingDataset(level={self.target_level}, samples={len(self)}, "
            f"classes={len(self.classes)})"
        )


def _preview(dataset: CoralTrainingDataset, num_samples: int = 3, fetch: bool = False) -> None:
    print("========================================")
    print(repr(dataset))
    print(f"Classes (count={len(dataset.classes)}): {dataset.classes[:min(10, len(dataset.classes))]}")
    for i in range(min(num_samples, len(dataset))):
        path, target_idx = dataset.samples[i]
        target_name = dataset.idx_to_class[target_idx]
        print(f"[{i}] {path} -> {target_idx} ({target_name}) | caption='{dataset.captions[i]}'")
    if fetch:
        print("\nFetching items via __getitem__()...")
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                img = sample["image"]
                img_type = type(img).__name__
                print(
                    f"getitem[{i}] -> (image type: {img_type}, target: {sample['target']}, "
                    f"target_name: {sample['target_name']}, text: '{sample['text']}')"
                )
            except Exception as exc:
                print(f"getitem[{i}] failed: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview CoralTrainingDataset samples")
    parser.add_argument("--csv", default='/home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/reefnet.ai/output/cotw_species_images_flat_tax.csv',help="Path to annotations CSV")
    parser.add_argument("--data_root", default=None, help="Optional base directory for relative paths")
    parser.add_argument("--target_level", default="species", help="Taxonomy level for targets")
    parser.add_argument("--rank_only", action="store_true", help="Use rank-only labels for targets")
    parser.add_argument("--caption_level", default=None, help="Taxonomy level for captions (default: target level)")
    parser.add_argument("--caption_mode", default="chain", choices=["chain", "rank_only"], help="Caption mode")
    parser.add_argument("--caption_prefix", default=None, help="Optional prefix for every caption")
    parser.add_argument("--no_metadata", action="store_true", help="Disable returning metadata in __getitem__")
    parser.add_argument("--num_samples", type=int, default=3, help="How many samples to preview")
    parser.add_argument("--fetch", action="store_true", help="Load samples via __getitem__ for inspection")
    parser.add_argument("--path_key", default=None, help="Explicit column to use for image paths")
    parser.add_argument("--no_normalize_anthozoa", action="store_true", help="Disable Anthozoa normalization")
    parser.add_argument("--keep-missing-targets", action="store_true", help="Retain rows without the requested taxonomy rank.")
    parser.add_argument("--no-infer-genus-from-species", action="store_true", help="Do not infer genus from binomial species text.")
    parser.add_argument("--path-replace-from", default=None, help="Optional path prefix/string to replace.")
    parser.add_argument("--path-replace-to", default=None, help="Replacement for --path-replace-from.")

    args = parser.parse_args()

    dataset = CoralTrainingDataset(
        csv_file=args.csv,
        data_root=args.data_root,
        target_level=args.target_level,
        rank_only=args.rank_only,
        caption_level=args.caption_level,
        caption_mode=args.caption_mode,
        caption_prefix=args.caption_prefix,
        normalize_anthozoa=not args.no_normalize_anthozoa,
        path_key=args.path_key,
        drop_missing_targets=not args.keep_missing_targets,
        infer_genus_from_species=not args.no_infer_genus_from_species,
        path_replace_from=args.path_replace_from,
        path_replace_to=args.path_replace_to,
        return_metadata=not args.no_metadata,
    )

    if args.keep_missing_targets:
        missing_targets = sum(1 for idx in dataset.target_indices if idx < 0)
        print(f"Missing target rows kept: {missing_targets}")
        if missing_targets:
            first_missing = next(i for i, idx in enumerate(dataset.target_indices) if idx < 0)
            print(
                "Example missing-target sample:"
                f" path='{dataset.paths[first_missing]}',"
                f" caption='{dataset.captions[first_missing]}'"
            )
    _preview(dataset, num_samples=args.num_samples, fetch=args.fetch)
