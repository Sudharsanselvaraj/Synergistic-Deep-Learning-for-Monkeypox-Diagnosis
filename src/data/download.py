"""Download the raw datasets for the Tri-Net MPOX rebuild from Kaggle.

The paper's 14-class lesion set is a *composite*. We reconstruct it honestly from:
  - MSLD v2    : 6 pox/infectious classes (Monkeypox, Chickenpox, Cowpox, HFMD, Healthy, Measles)
  - ISIC 9-cls : 8 dermoscopy classes     (ActinicKeratosis, BCC, Dermatofibroma, Melanoma,
                 MelanocyticNevi, BenignKeratosis, VascularLesion, SquamousCellCarcinoma)
  - symptom    : the Kaggle monkeypox symptom CSV (binary module)

Prereq: a Kaggle API token at ~/.kaggle/kaggle.json  (Kaggle → Account → Create New API Token).

Usage:
    python -m src.data.download            # all datasets
    python -m src.data.download --only msld_v2 symptom
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CFG, ensure_dirs  # noqa: E402

# name -> (kaggle slug, is_competition?)  — datasets only here
DATASETS = {
    "msld_v2":    "joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20",  # 6 pox classes, ~595 MB
    "dermoscopy": "nodoubttome/skin-cancer9-classesisic",                     # 8 derm classes, ~1.6 GB
    "symptom":    "shuvoalok/monkeypox-dataset",                              # symptom CSV, ~300 KB
}


def _check_kaggle() -> None:
    import os
    kdir = Path.home() / ".kaggle"
    classic = kdir / "kaggle.json"           # classic {username,key}
    access = kdir / "access_token"           # newer KGAT_ token
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"))
    if not (classic.exists() or access.exists() or has_env):
        sys.exit(
            "\n[!] No Kaggle credentials found.\n"
            "    kaggle.com → Settings → 'Create New API Token', then either:\n"
            "      • classic: save kaggle.json to ~/.kaggle/kaggle.json, or\n"
            "      • new token: printf '%s' 'KGAT_...' > ~/.kaggle/access_token\n"
            "    then chmod 600 the file.\n"
        )
    for f in (classic, access):
        if f.exists():
            f.chmod(0o600)


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def download_one(api, key: str) -> None:
    slug = DATASETS[key]
    dest = CFG.data_raw / key
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"[=] {key}: already present at {dest} — skipping download.")
        return
    print(f"[>] {key}: downloading {slug} ...")
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    print(f"[✓] {key}: ready at {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(DATASETS), help="subset of datasets")
    args = ap.parse_args()
    ensure_dirs()
    _check_kaggle()
    api = _api()
    keys = args.only or list(DATASETS)
    for k in keys:
        download_one(api, k)
    print("\nAll requested datasets downloaded. Next: python -m src.data.prepare_lesion")


if __name__ == "__main__":
    main()
