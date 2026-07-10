"""Assemble the 14-class lesion dataset and make a clean, leakage-free 80/10/10 split.

Sources (see PROJECT_PLAN.md):
  - MSLD v2 : 6 pox/infectious classes  (ORIGINAL images only — we do NOT touch MSLD's
              pre-augmented set, because augmented copies of a training image leaking into
              val/test is exactly what inflates naive results. We augment at TRAIN time only.)
  - ISIC    : 8 dermoscopy classes

Output: data/processed/{train,val,test}/<ClassName>/*.jpg  (class folders Keras can read),
plus a manifest.csv and a printed count table. Stratified, seeded, split on unique images.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from trinet.config import CFG, ensure_dirs  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# target class name -> list of source folder-name aliases (case-insensitive match on dir name)
POX_MAP = {
    "Mpox": ["Monkeypox"],
    "Chickenpox": ["Chickenpox"],
    "Cowpox": ["Cowpox"],
    "Measles": ["Measles"],
    "HFMD": ["HFMD"],
    "Healthy": ["Healthy"],
}
DERM_MAP = {
    "Melanoma": ["melanoma"],
    "MelanocyticNevi": ["nevus"],
    "BasalCellCarcinoma": ["basal cell carcinoma"],
    "ActinicKeratosis": ["actinic keratosis"],
    "BenignKeratosis": ["pigmented benign keratosis"],
    "Dermatofibroma": ["dermatofibroma"],
    "VascularLesion": ["vascular lesion"],
    "SquamousCellCarcinoma": ["squamous cell carcinoma"],
}


def _find_class_dirs(root: Path, aliases: list[str]) -> list[Path]:
    """All directories under `root` whose name matches an alias (case-insensitive)."""
    wanted = {a.lower() for a in aliases}
    return [d for d in root.rglob("*") if d.is_dir() and d.name.lower() in wanted]


def _collect(root: Path, aliases: list[str], exclude_aug: bool) -> list[Path]:
    files: list[Path] = []
    for d in _find_class_dirs(root, aliases):
        # skip MSLD's augmented tree to avoid train/val/test leakage of derived images
        if exclude_aug and "aug" in str(d).lower():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in IMG_EXT:
                files.append(f)
    # de-dup by filename (MSLD repeats the same originals across folds)
    seen, uniq = set(), []
    for f in sorted(files):
        if f.name not in seen:
            seen.add(f.name)
            uniq.append(f)
    return uniq


def _split(files: list[Path], rng: np.random.Generator) -> dict[str, list[Path]]:
    idx = rng.permutation(len(files))
    n = len(files)
    n_tr = int(round(CFG.train_frac * n))
    n_va = int(round(CFG.val_frac * n))
    tr = [files[i] for i in idx[:n_tr]]
    va = [files[i] for i in idx[n_tr : n_tr + n_va]]
    te = [files[i] for i in idx[n_tr + n_va :]]
    return {"train": tr, "val": va, "test": te}


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(CFG.seed)
    msld_root = CFG.data_raw / "msld_v2"
    derm_root = CFG.data_raw / "dermoscopy"

    if CFG.data_proc.exists():
        shutil.rmtree(CFG.data_proc)

    sources = [(POX_MAP, msld_root, True), (DERM_MAP, derm_root, False)]
    manifest_rows = ["path,class,split"]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for cmap, root, excl_aug in sources:
        for cls, aliases in cmap.items():
            files = _collect(root, aliases, exclude_aug=excl_aug)
            if not files:
                print(f"[!] {cls}: NO images found under {root} (aliases={aliases})")
                continue
            for split, flist in _split(files, rng).items():
                dst_dir = CFG.data_proc / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src in flist:
                    dst = dst_dir / f"{cls}_{src.name}"
                    shutil.copy2(src, dst)
                    manifest_rows.append(f"{dst.relative_to(CFG.root)},{cls},{split}")
                counts[cls][split] = len(flist)

    (CFG.data_proc / "manifest.csv").write_text("\n".join(manifest_rows))

    # print the honest count table
    print(f"\n{'class':<22}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    print("-" * 48)
    tot = {"train": 0, "val": 0, "test": 0}
    for cls in CFG.lesion_classes:
        c = counts[cls]
        t = c["train"] + c["val"] + c["test"]
        for k in tot:
            tot[k] += c[k]
        print(f"{cls:<22}{c['train']:>7}{c['val']:>6}{c['test']:>6}{t:>7}")
    print("-" * 48)
    grand = sum(tot.values())
    print(f"{'TOTAL':<22}{tot['train']:>7}{tot['val']:>6}{tot['test']:>6}{grand:>7}")
    print(f"\n[✓] processed split written to {CFG.data_proc}")
    print("    next: python -m src.models.extract_features")


if __name__ == "__main__":
    main()
