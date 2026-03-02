from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.imageomics.naming_eval import Taxon


class FranCatalogueLoader(Dataset):
    """
    General CSV-based dataset loader for Fran catalogue annotations.

    Expected CSV columns (at least):
      - Path: absolute or relative path to image
      - kingdom, phylum, class, order, family, genus, species (taxonomy)
      - Optional: ScientificName

    Parameters
    - csv_file: path to the CSV file containing annotations
    - data_root: base directory to resolve relative paths in 'Path' (optional)
    - taxonomic_level: which level to evaluate at: one of
        ['kingdom','phylum','cls','order','family','genus','species','scientific']
        Note: 'cls' is the canonical key for the "class" taxonomic rank to
        avoid clashing with Python keyword in downstream tools.
    - transform: torchvision-style transform to apply to loaded PIL image

    Behavior
    - Builds label indices from the chosen taxonomic level.
    - Exposes .classes (list[str]) and .class_to_idx (dict[str,int]).
    - __getitem__ returns (image_tensor, target_idx) compatible with eval code.
    - .samples is a list of (filepath, class_idx) for debugging.
    """

    def __init__(
        self,
        csv_file: str,
        data_root: Optional[str] = None,
        taxonomic_level: str = "genus",
        transform=None,
        require_species: bool = False,
        rank_only: bool = False,
        normalize_anthozoa: bool = True,
        taxonomy_filters: Optional[dict[str, str]] = None,
    ) -> None:
        if not os.path.isabs(csv_file) and data_root is not None:
            csv_file = os.path.join(data_root, csv_file)

        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data_root = data_root

        # Normalize column names to lower for robust access
        cols = {c.lower(): c for c in self.data.columns}

        # Remap taxonomy 'class' column to 'cls' key used in other utils
        if "class" in cols and "cls" not in cols:
            self.data.rename(columns={cols["class"]: "cls"}, inplace=True)
            cols = {c.lower(): c for c in self.data.columns}

        # Optional exact-match filtering on taxonomy columns (case-insensitive)
        if taxonomy_filters:
            mask = pd.Series([True] * len(self.data))
            for key, value in taxonomy_filters.items():
                if value is None:
                    continue
                lookup = key.lower()
                if lookup == "class":
                    lookup = "cls"
                col = cols.get(lookup)
                if col is None:
                    raise KeyError(f"taxonomy_filters refers to unknown column '{key}'")
                series = self.data[col].astype(str).str.strip().str.lower()
                mask &= series == str(value).strip().lower()
            self.data = self.data[mask].reset_index(drop=True)
            cols = {c.lower(): c for c in self.data.columns}

        # Determine label string per row according to the requested level
        taxonomic_level = taxonomic_level.lower()
        self.taxonomic_level = taxonomic_level

        levels_order = ["kingdom", "phylum", "cls", "order", "family", "genus", "species"]

        def to_taxon(row: pd.Series) -> Taxon:
            def getv(k):
                col = cols.get(k, k)
                return str(row[col]).strip() if col in row and pd.notna(row[col]) else ""

            kingdom = getv("kingdom")
            phylum = getv("phylum")
            cls_val = getv("cls") or getv("class")
            order = getv("order")
            family = getv("family")
            genus = getv("genus")
            species_raw = getv("species")

            # Normalize coral taxonomy: in TreeOfLife, Class=Anthozoa, Subclass=Hexacorallia.
            # If our CSV has Hexacorallia in the class column under Cnidaria, replace with Anthozoa
            if normalize_anthozoa:
                if (phylum.lower() == "cnidaria") and (cls_val.lower() == "hexacorallia"):
                    cls_val = "Anthozoa"

            # Parse species: accept binomial, avoid genus duplication
            species_epithet = ""
            if species_raw:
                toks = species_raw.split()
                if len(toks) >= 2:
                    if genus and toks[0].lower() == genus.lower():
                        species_epithet = toks[1]
                    else:
                        if not genus:
                            genus = toks[0]
                        species_epithet = toks[1]
                else:
                    species_epithet = toks[0]

            return Taxon(
                kingdom=kingdom,
                phylum=phylum,
                cls=cls_val,
                order=order,
                family=family,
                genus=genus,
                species=species_epithet,
            )

        # Optional filtering: require species epithet for species-level eval
        if require_species and taxonomic_level == "species":
            keep_mask = self.data.apply(lambda r: bool(to_taxon(r).species), axis=1)
            self.data = self.data[keep_mask].reset_index(drop=True)

        if taxonomic_level == "scientific":
            if rank_only:
                # rank_only for scientific -> just the binomial (same as scientific_name)
                labels = self.data.apply(lambda r: to_taxon(r).scientific_name or "", axis=1)
            else:
                labels = self.data.apply(lambda r: to_taxon(r).scientific_name or "", axis=1)
        elif taxonomic_level in set(levels_order):
            cutoff_idx = levels_order.index(taxonomic_level)

            if rank_only:
                def rank_label(row):
                    t = to_taxon(row)
                    tpl = t.to_tuple()
                    if cutoff_idx == 6:  # species -> return binomial
                        return t.scientific_name or ""
                    return tpl[cutoff_idx] if len(tpl) > cutoff_idx else ""
                labels = self.data.apply(rank_label, axis=1)
            else:
                def taxon_string(row):
                    t = to_taxon(row)
                    tpl = t.to_tuple()  # (Kingdom, Phylum, Class, Order, Family, Genus, species)
                    parts = [p for i, p in enumerate(tpl) if i <= cutoff_idx and p]
                    return " ".join(parts).strip()

                labels = self.data.apply(taxon_string, axis=1)
        else:
            raise ValueError("Unsupported taxonomic_level: %s" % taxonomic_level)

        # Resolve image paths
        path_key = cols.get("path", "Path")
        if path_key not in self.data:
            raise KeyError("CSV must contain a 'Path' column for image paths.")

        paths = self.data[path_key].astype(str).tolist()
        if self.data_root is not None:
            paths = [p if os.path.isabs(p) else os.path.join(self.data_root, p) for p in paths]
        self.paths = paths

        # Build label mapping
        self.labels = labels.tolist()
        unique_classes = sorted({lbl for lbl in self.labels if isinstance(lbl, str) and len(lbl) > 0})
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.classes = unique_classes

        # Vectorized mapping to indices
        self.targets = [self.class_to_idx[lbl] for lbl in self.labels]

        # Expose samples list useful for downstream few-shot debug
        self.samples: list[Tuple[str, int]] = list(zip(self.paths, self.targets))

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"FranCatalogue (level={self.taxonomic_level}, samples={len(self)}, classes={len(self.classes)})"

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

    parser = argparse.ArgumentParser(description="Test FranCatalogueLoader across taxonomy levels")
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/fran_cat_annotations.csv",
        help="Path to fran_cat_annotations.csv",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base dir to resolve relative paths in CSV (optional)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=["kingdom","phylum","cls","order","family","genus","species","scientific"],
        help="Taxonomy levels to test",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="If set, actually loads a few images via __getitem__ to show return types",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="How many samples to preview",
    )
    parser.add_argument(
        "--normalize_anthozoa",
        action="store_true",
        help="If set, when phylum=Cnidaria and class=Hexacorallia, replace class with Anthozoa to match TreeOfLife taxonomy.",
    )
    parser.add_argument(
        "--require_species",
        action="store_true",
        help="Drop rows without species epithet when taxonomic_level is 'species'",
    )

    args = parser.parse_args()

    def preview(loader: FranCatalogueLoader, n: int = 3, fetch: bool = False):
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
        ds = FranCatalogueLoader(
            csv_file=args.csv,
            data_root=args.data_root,
            taxonomic_level=lvl,
            transform=None,
            require_species=args.require_species,
            normalize_anthozoa=args.normalize_anthozoa,
        )
        preview(ds, n=args.num_samples, fetch=args.fetch)
