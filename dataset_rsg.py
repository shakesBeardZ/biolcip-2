from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.imageomics.naming_eval import Taxon


class RSGPatchDataset(Dataset):
    """
    Patch-based dataset loader for RSG data, aligned with FranCatalogueLoader.

    Expects a CSV of patches with at least:
      - patch_path: path to image patch
      - Our_Labels: genus label for the patch

    Additionally loads a taxonomy lookup CSV (new_fran_catalogue.csv) to map each genus
    to the full taxonomy (kingdom..species). Species may be empty for most RSG patches.

    Parameters
    - csv_file: path to RSG patches CSV
    - taxonomy_csv: path to new_fran_catalogue.csv providing taxonomy columns
    - taxonomic_level: ['kingdom','phylum','cls','order','family','genus','species','scientific']
    - transform: torchvision-style transform
    - require_species: if True and level=='species', drop items with empty species epithet
    - rank_only: if True, label is exactly the rank value (species -> binomial)
    - normalize_anthozoa: if True, when phylum=Cnidaria and cls=Hexacorallia, set cls='Anthozoa'
    - corals_only: if True, filter to a predefined set of coral genera used in RSG

    Exposes
    - classes, class_to_idx, idx_to_class, labels, targets, samples, nb_classes, class_counts
    - __getitem__ returns (image, target)
    """

    CORAL_GENERA = {
        "Porites","Acropora","Pocillopora","Montipora","Goniastrea",
        "Echinopora","Stylophora","Favites","Lobophyllia","Seriatopora",
        "Galaxea","Astreopora","Tubastraea","Plerogyra"
    }

    def __init__(
        self,
        csv_file: str,
        taxonomy_csv: str = "/home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/new_fran_catalogue.csv",
        taxonomic_level: str = "genus",
        transform=None,
        require_species: bool = False,
        rank_only: bool = False,
        normalize_anthozoa: bool = True,
        corals_only: bool = False,
        path_key: Optional[str] = None,
        label_key: Optional[str] = None,
    ) -> None:
        # Load patches CSV
        df = pd.read_csv(csv_file)

        # Resolve column names
        path_key = path_key or ("patch_path" if "patch_path" in df.columns else ("Path" if "Path" in df.columns else None))
        label_key = label_key or ("Our_Labels" if "Our_Labels" in df.columns else ("genus" if "genus" in df.columns else ("Experimental_label" if "Experimental_label" in df.columns else None)))
        if path_key is None or label_key is None:
            raise KeyError("CSV must contain 'patch_path' (or 'Path') and 'Our_Labels' (or 'genus'/'Experimental_label') columns")

        # Optionally restrict to coral genera used in RSG
        if corals_only:
            df = df[df[label_key].isin(self.CORAL_GENERA)].reset_index(drop=True)

        # Load taxonomy lookup CSV and normalize columns
        tax = pd.read_csv(taxonomy_csv)
        # Normalize columns to lower and rename class->cls
        tax_cols = {c.lower(): c for c in tax.columns}
        if "class" in tax_cols and "cls" not in tax_cols:
            tax.rename(columns={tax_cols["class"]: "cls"}, inplace=True)
            tax_cols = {c.lower(): c for c in tax.columns}

        # Ensure expected columns exist in taxonomy CSV
        expected = ["kingdom","phylum","cls","order","family","genus","species"]
        missing = [c for c in expected if c not in {k.lower() for k in tax.columns}]
        if missing:
            raise KeyError(f"taxonomy_csv missing columns: {missing}")

        # Reduce taxonomy to one row per genus (take first occurrence)
        tax_norm = tax.rename(columns={tax_cols.get(k, k): k for k in expected})
        tax_norm_cols = ["kingdom","phylum","cls","order","family","genus","species"]
        tax_norm = tax_norm[tax_norm_cols].copy()
        tax_norm["genus_l"] = tax_norm["genus"].astype(str).str.strip().str.lower()
        tax_dedup = tax_norm.drop_duplicates(subset=["genus_l"]).set_index("genus_l")

        # Assemble dataset rows with resolved taxonomy from genus
        rows: List[dict] = []
        for _, r in df.iterrows():
            path = str(r[path_key])
            genus = str(r[label_key]).strip()
            genus_l = genus.lower()
            if genus_l in tax_dedup.index:
                rec = tax_dedup.loc[genus_l]
                kingdom = str(rec["kingdom"]).strip()
                phylum = str(rec["phylum"]).strip()
                cls_val = str(rec["cls"]).strip()
                order = str(rec["order"]).strip()
                family = str(rec["family"]).strip()
                genus_res = str(rec["genus"]).strip() or genus
                species_ep = str(rec["species"]).strip()
            else:
                # Fallback: only genus known
                kingdom = phylum = order = family = species_ep = ""
                cls_val = ""
                genus_res = genus

            if normalize_anthozoa and phylum.lower() == "cnidaria" and cls_val.lower() == "hexacorallia":
                cls_val = "Anthozoa"

            rows.append({
                "Path": path,
                "kingdom": kingdom,
                "phylum": phylum,
                "cls": cls_val,
                "order": order,
                "family": family,
                "genus": genus_res,
                "species": species_ep,
            })

        # Build DataFrame and labels like FranCatalogueLoader
        self.data = pd.DataFrame(rows)
        self.transform = transform
        taxonomic_level = taxonomic_level.lower()
        self.taxonomic_level = taxonomic_level

        levels_order = ["kingdom", "phylum", "cls", "order", "family", "genus", "species"]

        def to_taxon(row: pd.Series) -> Taxon:
            t = Taxon(
                kingdom=str(row["kingdom"]).strip(),
                phylum=str(row["phylum"]).strip(),
                cls=str(row["cls"]).strip(),
                order=str(row["order"]).strip(),
                family=str(row["family"]).strip(),
                genus=str(row["genus"]).strip(),
                species=str(row["species"]).strip(),
            )
            if normalize_anthozoa:
                if t.phylum.lower() == "cnidaria" and t.cls.lower() == "hexacorallia":
                    t = Taxon(kingdom=t.kingdom, phylum=t.phylum, cls="Anthozoa", order=t.order, family=t.family, genus=t.genus, species=t.species)
            return t

        # Optional species filtering
        data_df = self.data
        if require_species and taxonomic_level == "species":
            keep_mask = data_df.apply(lambda r: bool(to_taxon(r).species), axis=1)
            data_df = data_df[keep_mask].reset_index(drop=True)

        # Labels
        if taxonomic_level == "scientific":
            labels = data_df.apply(lambda r: to_taxon(r).scientific_name or "", axis=1)
        elif taxonomic_level in set(levels_order):
            cutoff_idx = levels_order.index(taxonomic_level)
            if rank_only:
                def rank_label(row):
                    t = to_taxon(row)
                    tpl = t.to_tuple()
                    if cutoff_idx == 6:
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
            raise ValueError(f"Unsupported taxonomic_level: {taxonomic_level}")

        # Paths and mapping
        self.paths = data_df["Path"].astype(str).tolist()
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
        self.class_counts = [0] * self.nb_classes
        for t in self.targets:
            self.class_counts[t] += 1

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"RSGPatch (level={self.taxonomic_level}, samples={len(self)}, classes={len(self.classes)})"

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

    parser = argparse.ArgumentParser(description="Test RSGPatchDataset across taxonomy levels")
    parser.add_argument("--csv", type=str, required=True, help="Path to RSG patches CSV")
    parser.add_argument("--taxonomy_csv", type=str, default="/home/yahiab/reefnet_project/yahia_code/data_preprocessing_scripts/data/new_fran_catalogue.csv", help="Path to taxonomy CSV")
    parser.add_argument("--levels", type=str, nargs="+", default=["kingdom","phylum","cls","order","family","genus","species","scientific"], help="Taxonomy levels to test")
    parser.add_argument("--fetch", action="store_true", help="Load a few samples via __getitem__")
    parser.add_argument("--num_samples", type=int, default=3, help="How many samples to preview")
    parser.add_argument("--normalize_anthozoa", action="store_true", help="Normalize cls=Hexacorallia to Anthozoa under Cnidaria")
    parser.add_argument("--require_species", action="store_true", help="Drop rows without species when level is 'species'")
    parser.add_argument("--corals_only", action="store_true", help="Filter to the predefined coral genera")

    args = parser.parse_args()

    def preview(loader: RSGPatchDataset, n: int = 3, fetch: bool = False):
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
        ds = RSGPatchDataset(
            csv_file=args.csv,
            taxonomy_csv=args.taxonomy_csv,
            taxonomic_level=lvl,
            transform=None,
            require_species=args.require_species,
            normalize_anthozoa=args.normalize_anthozoa,
            corals_only=args.corals_only,
        )
        preview(ds, n=args.num_samples, fetch=args.fetch)
