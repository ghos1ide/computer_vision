"""Inspect VOC-style segmentation masks for class distribution and visualization."""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

VOC_SUBDIR = Path("VOCdevkit") / "VOC2012"


def _voc_palette() -> List[int]:
    base = [
        0, 0, 0,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        0, 0, 128,
        128, 0, 128,
        0, 128, 128,
        128, 128, 128,
        64, 0, 0,
        192, 0, 0,
        64, 128, 0,
        192, 128, 0,
        64, 0, 128,
        192, 0, 128,
        64, 128, 128,
        192, 128, 128,
        0, 64, 0,
        128, 64, 0,
        0, 192, 0,
        128, 192, 0,
        0, 64, 128,
    ]
    palette = [0] * (256 * 3)
    for idx in range(len(base) // 3):
        start = idx * 3
        palette[start:start + 3] = base[start:start + 3]
    palette[255 * 3:255 * 3 + 3] = [255, 0, 255]
    return palette


def _load_ids(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Split file is empty: {split_file}")
    return ids


def _save_mask(mask: np.ndarray, save_path: Path) -> None:
    mask_u8 = mask.astype(np.uint8)
    mask_img = Image.fromarray(mask_u8, mode="P")
    mask_img.putpalette(_voc_palette())
    mask_img.save(save_path)


def analyze_masks(
    data_root: Path,
    split: str,
    num_classes: int,
    ignore_index: int,
    max_samples: int,
    save_dir: Optional[Path],
    per_sample: bool,
) -> None:
    split_root = data_root / "splits"
    split_file = split_root / f"{split}.txt"
    ids = _load_ids(split_file)
    if max_samples > 0:
        ids = ids[:max_samples]

    voc_root = data_root / VOC_SUBDIR
    mask_dir = voc_root / "SegmentationClass"
    image_dir = voc_root / "JPEGImages"

    total_counts = np.zeros(num_classes, dtype=np.int64)
    total_valid = 0
    total_ignore = 0
    total_fg = 0
    samples_with_fg = 0

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in ids:
        mask_path = mask_dir / f"{sample_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = np.array(Image.open(mask_path), dtype=np.int64)
        valid = mask != ignore_index
        valid_count = int(valid.sum())
        ignore_count = int((~valid).sum())
        total_valid += valid_count
        total_ignore += ignore_count

        if valid_count > 0:
            hist = np.bincount(mask[valid].ravel(), minlength=num_classes)
            total_counts += hist[:num_classes]

        fg = (mask > 0) & valid
        fg_count = int(fg.sum())
        total_fg += fg_count
        if fg_count > 0:
            samples_with_fg += 1

        if per_sample:
            unique_vals = np.unique(mask).tolist()
            fg_ratio = (fg_count / valid_count) if valid_count > 0 else 0.0
            print(
                f"[sample] {sample_id} unique={unique_vals} "
                f"valid={valid_count} ignore={ignore_count} fg_ratio={fg_ratio:.4f}"
            )

        if save_dir is not None:
            _save_mask(mask, save_dir / f"{sample_id}_mask.png")
            image_path = image_dir / f"{sample_id}.jpg"
            if image_path.exists():
                Image.open(image_path).convert("RGB").save(save_dir / f"{sample_id}_image.jpg")

    if total_valid == 0:
        print("[summary] no valid pixels found (all ignored).")
        return

    fg_ratio = total_fg / total_valid
    print(
        f"[summary] split={split} samples={len(ids)} valid={total_valid} "
        f"ignore={total_ignore} fg_ratio={fg_ratio:.4f} samples_with_fg={samples_with_fg}"
    )

    print("[summary] class distribution (non-ignore):")
    for cls in range(num_classes):
        count = int(total_counts[cls])
        if count <= 0:
            continue
        ratio = count / total_valid
        print(f"  class={cls:02d} count={count} ratio={ratio:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect segmentation masks and class distribution.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root for VOC and split files.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--save-dir", type=str, default="", help="Optional directory to save masks/images.")
    parser.add_argument("--per-sample", action="store_true", help="Print per-sample stats.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else None
    analyze_masks(
        data_root=Path(args.data_root),
        split=args.split,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_samples=args.max_samples,
        save_dir=save_dir,
        per_sample=args.per_sample,
    )


if __name__ == "__main__":
    main()
