"""Prepare PASCAL VOC2012 segmentation dataset and splits."""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Union

VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_ARCHIVE_NAME = "VOCtrainval_11-May-2012.tar"
VOC_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
VOC_SUBDIR = Path("VOCdevkit") / "VOC2012"


def _read_split_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Split file is empty: {path}")
    return ids


def _write_split_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")


def ensure_voc_download(data_root: Path) -> Path:
    from jittor_utils.misc import download_and_extract_archive

    voc_root = data_root / VOC_SUBDIR
    if voc_root.exists():
        print(f"[prepare] VOC2012 already exists: {voc_root}")
        return voc_root

    data_root.mkdir(parents=True, exist_ok=True)
    print("[prepare] Downloading VOC2012 train/val archive...")
    download_and_extract_archive(
        VOC_URL,
        str(data_root),
        filename=VOC_ARCHIVE_NAME,
        md5=VOC_MD5,
    )

    if not voc_root.exists():
        raise RuntimeError(f"VOC2012 was not extracted as expected: {voc_root}")

    print(f"[prepare] Downloaded and extracted VOC2012 to: {voc_root}")
    return voc_root


def create_train_val_test_splits(
    voc_root: Path,
    split_root: Path,
    val_ratio: float = 0.5,
    seed: int = 42,
    force: bool = False,
) -> Path:
    if not (0.05 <= val_ratio < 1.0):
        raise ValueError("val_ratio must satisfy 0.05 <= val_ratio < 1.0")

    train_split = split_root / "train.txt"
    val_split = split_root / "val.txt"
    test_split = split_root / "test.txt"

    if train_split.exists() and val_split.exists() and test_split.exists() and not force:
        print(f"[prepare] Existing splits found in: {split_root}")
        return split_root

    official_split_root = voc_root / "ImageSets" / "Segmentation"
    official_train = _read_split_ids(official_split_root / "train.txt")
    official_val = _read_split_ids(official_split_root / "val.txt")

    if len(official_val) < 2:
        raise RuntimeError("Official VOC val split has fewer than 2 samples; cannot create val/test.")

    val_candidates = official_val[:]
    rng = random.Random(seed)
    rng.shuffle(val_candidates)

    val_count = int(round(len(val_candidates) * val_ratio))
    val_count = max(1, min(val_count, len(val_candidates) - 1))

    custom_val = sorted(val_candidates[:val_count])
    custom_test = sorted(val_candidates[val_count:])

    split_root.mkdir(parents=True, exist_ok=True)
    _write_split_ids(train_split, sorted(official_train))
    _write_split_ids(val_split, custom_val)
    _write_split_ids(test_split, custom_test)

    print("[prepare] Created custom segmentation splits:")
    print(f"          train: {len(official_train)} samples (official train)")
    print(f"          val  : {len(custom_val)} samples (from official val)")
    print(f"          test : {len(custom_test)} samples (from official val)")
    return split_root


def prepare_voc_dataset(
    data_root: Union[str, Path],
    val_ratio: float = 0.5,
    seed: int = 42,
    force_split: bool = False,
) -> Tuple[Path, Path]:
    root = Path(data_root).resolve()
    voc_root = ensure_voc_download(root)
    split_root = root / "splits"
    create_train_val_test_splits(
        voc_root=voc_root,
        split_root=split_root,
        val_ratio=val_ratio,
        seed=seed,
        force=force_split,
    )
    return voc_root, split_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare VOC2012 segmentation dataset and splits.")
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory for dataset files.")
    parser.add_argument("--val-ratio", type=float, default=0.5, help="Ratio of official val used as custom val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Regenerate train/val/test split files even if they already exist.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prepare_voc_dataset(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_split=args.force_split,
    )


if __name__ == "__main__":
    main()
