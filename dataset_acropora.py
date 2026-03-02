from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.imageomics.naming_eval import Taxon


class AcroporaDataset(Dataset):
    """
    Folder-based dataset loader for the Acropora-only dataset, modeled after FranCatalogueLoader.

    Assumptions
    - All images belong to genus 'Acropora'.
    - Each subfolder under the given root represents a species (folder name may be
      'Acropora_xxx', 'Acropora xxx', or just the species epithet with underscores).

    Parameters
    - data_root: base directory containing subfolders of species for the selected split.
                 If not provided, defaults are used based on 'split'.
    - split: 'train' or 'val'. Used only to choose default roots and optional exclusions.
    - taxonomic_level: one of ['kingdom','phylum','cls','order','family','genus','species','scientific'].
    - transform: torchvision-style transform to apply to loaded PIL image.
    - require_species: if True and taxonomic_level=='species', drop images missing species epithet.
    - rank_only: if True, produce labels only for the selected rank (species -> binomial).
    - normalize_anthozoa: if True and (phylum=Cnidaria & cls=Hexacorallia), replace cls with 'Anthozoa'.
    - exclude_folders: optional list of folder names to skip (useful for validation exclusions).

    Exposes
    - classes, class_to_idx, idx_to_class, labels, targets, samples
    - __getitem__ returns (image_tensor_or_PIL, target_idx)
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = "train",
        taxonomic_level: str = "genus",
        transform=None,
        require_species: bool = False,
        rank_only: bool = False,
        normalize_anthozoa: bool = True,
        exclude_folders: Optional[List[str]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif", ".JPG", ".PNG"),
    ) -> None:
        # Default roots from the original script if not provided
        default_train_root = "/home/yahiab/reefnet_project/Acropora_dataset/V2/Acropora species database/"
        default_val_root = "/home/yahiab/reefnet_project/Acropora_dataset/RSG_Acropora_Identification/AI_Acropora_Database"

        self.split = split.lower()
        if data_root is None:
            if self.split == "val":
                data_root = default_val_root
            else:
                data_root = default_train_root
        self.data_root = data_root

        # Default exclusions for validation as per original script
        if exclude_folders is None and self.split == "val":
            exclude_folders = ["Not_Acropora", "Acropora_not_confirmed", "Image_Quality_Issue"]
        self.exclude_folders = set(exclude_folders or [])

        self.transform = transform
        taxonomic_level = taxonomic_level.lower()
        self.taxonomic_level = taxonomic_level
        self.require_species = require_species
        self.rank_only = rank_only
        self.normalize_anthozoa = normalize_anthozoa

        # Collect file list and minimal taxonomy table
        rows = self._scan_folders(self.data_root, extensions)
        if len(rows) == 0:
            raise RuntimeError(f"No images found under: {self.data_root}")

        # Build DataFrame with taxonomy columns similar to CSV loader
        self.data = pd.DataFrame(rows)
        # Ensure expected columns exist
        for col in ["Path", "kingdom", "phylum", "cls", "order", "family", "genus", "species"]:
            if col not in self.data.columns:
                self.data[col] = ""

        # Construct labels like in FranCatalogueLoader
        levels_order = ["kingdom", "phylum", "cls", "order", "family", "genus", "species"]

        def to_taxon(row: pd.Series) -> Taxon:
            kingdom = str(row["kingdom"]).strip()
            phylum = str(row["phylum"]).strip()
            cls_val = str(row["cls"]).strip() or str(row.get("class", "")).strip()
            order = str(row["order"]).strip()
            family = str(row["family"]).strip()
            genus = str(row["genus"]).strip()
            species_epithet = str(row["species"]).strip()

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
                species=species_epithet,
            )

        # Optional filtering for species level
        data_df = self.data
        if self.require_species and self.taxonomic_level == "species":
            keep_mask = data_df.apply(lambda r: bool(to_taxon(r).species), axis=1)
            data_df = data_df[keep_mask].reset_index(drop=True)

        # Build labels
        if self.taxonomic_level == "scientific":
            if self.rank_only:
                labels = data_df.apply(lambda r: to_taxon(r).scientific_name or "", axis=1)
            else:
                labels = data_df.apply(lambda r: to_taxon(r).scientific_name or "", axis=1)
        elif self.taxonomic_level in set(levels_order):
            cutoff_idx = levels_order.index(self.taxonomic_level)
            if self.rank_only:
                def rank_label(row):
                    t = to_taxon(row)
                    tpl = t.to_tuple()
                    if cutoff_idx == 6:  # species -> return binomial
                        return t.scientific_name or ""
                    return tpl[cutoff_idx] if len(tpl) > cutoff_idx else ""
                labels = data_df.apply(rank_label, axis=1)
            else:
                def taxon_string(row):
                    t = to_taxon(row)
                    tpl = t.to_tuple()
                    parts = [p for i, p in enumerate(tpl) if i <= cutoff_idx and p]
                    return " ".join(parts).strip()
                labels = data_df.apply(taxon_string, axis=1)
        else:
            raise ValueError(f"Unsupported taxonomic_level: {self.taxonomic_level}")

        # Resolve paths and build indices
        paths = data_df["Path"].astype(str).tolist()
        self.paths = paths
        self.labels = labels.tolist()
        unique_classes = sorted({lbl for lbl in self.labels if isinstance(lbl, str) and len(lbl) > 0})
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.classes = unique_classes
        self.targets = [self.class_to_idx.get(lbl, -1) for lbl in self.labels]
        if any(t < 0 for t in self.targets):
            raise RuntimeError("Some samples did not map to valid class indices. Check labels generation.")

        self.samples: List[Tuple[str, int]] = list(zip(self.paths, self.targets))
        self.nb_classes = len(self.classes)
        # class_counts aligned to class index order
        self.class_counts = [0] * self.nb_classes
        for t in self.targets:
            self.class_counts[t] += 1

    def _scan_folders(self, root_dir: str, extensions: Tuple[str, ...]) -> List[dict]:
        rows: List[dict] = []
        if not os.path.isdir(root_dir):
            return rows

        folders = sorted(os.listdir(root_dir))
        for folder in folders:
            if folder in self.exclude_folders:
                continue
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            species_epithet = self._parse_species_from_folder(folder)

            for fname in os.listdir(folder_path):
                if not fname.lower().endswith(tuple(ext.lower() for ext in extensions)):
                    continue
                fpath = os.path.join(folder_path, fname)
                if not os.path.isfile(fpath):
                    continue

                # Build taxonomy row; class defaults to Hexacorallia as per original note, may be normalized later
                rows.append({
                    "Path": fpath,
                    "kingdom": "Animalia",
                    "phylum": "Cnidaria",
                    "cls": "Hexacorallia",  # will be normalized to Anthozoa if requested
                    "order": "Scleractinia",
                    "family": "Acroporidae",
                    "genus": "Acropora",
                    "species": species_epithet,
                })
        return rows

    @staticmethod
    def _parse_species_from_folder(folder_name: str) -> str:
        # Replace underscores with spaces, trim
        name = folder_name.replace("_", " ").strip()
        if not name:
            return ""
        toks = name.split()
        if len(toks) == 1:
            # Single token; if it's not the genus, treat as epithet
            return "" if toks[0].lower() == "acropora" else toks[0]
        # If starts with genus, take epithet; else assume last token is epithet
        if toks[0].lower() == "acropora":
            return toks[1]
        # Some folders might be "Acropora something something"; take second token as epithet when genus present later
        try:
            idx = toks.index("Acropora")
            if idx + 1 < len(toks):
                return toks[idx + 1]
        except ValueError:
            pass
        # Fallback: second token
        return toks[1]

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"AcroporaFolder (level={self.taxonomic_level}, samples={len(self)}, classes={len(self.classes)})"

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        target = self.targets[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(img, torch.Tensor):
            if torch.isnan(img).any() or torch.isinf(img).any():
                raise ValueError(f"Invalid tensor for image at index {idx}: {path}")

        return img, target


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Test AcroporaDataset across taxonomy levels")
    parser.add_argument("--root", type=str, default=None, help="Data root (folder with species subfolders)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to use for defaults")
    parser.add_argument("--levels", type=str, nargs="+", default=["kingdom","phylum","cls","order","family","genus","species","scientific"], help="Taxonomy levels to test")
    parser.add_argument("--fetch", action="store_true", help="If set, actually loads a few images via __getitem__")
    parser.add_argument("--num_samples", type=int, default=3, help="How many samples to preview")
    parser.add_argument("--normalize_anthozoa", action="store_true", help="Normalize cls=Hexacorallia to Anthozoa under Cnidaria")
    parser.add_argument("--require_species", action="store_true", help="Drop rows without species epithet when level is 'species'")

    args = parser.parse_args()

    def preview(loader: AcroporaDataset, n: int = 3, fetch: bool = False):
        print("========================================")
        print(repr(loader))
        print(f"Classes (count={len(loader.classes)}):")
        pprint(loader.classes[:min(10, len(loader.classes))])
        print("Samples preview (path, class_idx, class_name):")
        for i in range(min(n, len(loader))):
            p = loader.paths[i]
            idx = loader.targets[i]
            name = loader.idx_to_class[idx]
            print(f"[{i}] {p} -> {idx} ({name})")
        if fetch and len(loader) > 0:
            print("\nFetching a couple of items via __getitem__...")
            for i in range(min(n, len(loader))):
                try:
                    img, target = loader[i]
                    itype = type(img).__name__
                    print(f"getitem[{i}] -> (image type: {itype}, target: {target}, class: {loader.idx_to_class[target]})")
                except Exception as e:
                    print(f"getitem[{i}] failed: {e}")

    for lvl in args.levels:
        print(f"\n>>> Testing taxonomic_level='{lvl}'")
        ds = AcroporaDataset(
            data_root=args.root,
            split=args.split,
            taxonomic_level=lvl,
            transform=None,
            require_species=args.require_species,
            normalize_anthozoa=args.normalize_anthozoa,
        )
        preview(ds, n=args.num_samples, fetch=args.fetch)

