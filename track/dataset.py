"""PASCAL VOC2012 segmentation dataset wrapper for Jittor."""

import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from jittor.dataset import Dataset


class VOCSegDataset(Dataset):
    """Custom VOC2012 segmentation dataset with synchronized image/mask transforms."""

    def __init__(
        self,
        voc_root: Union[str, Path],
        split_file: Union[str, Path],
        is_train: bool,
        use_augment: bool = True,
        crop_size: int = 320,
        base_size: int = 512,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.voc_root = Path(voc_root)
        self.split_file = Path(split_file)
        self.is_train = is_train
        self.use_augment = bool(use_augment)
        self.crop_size = crop_size
        self.base_size = base_size
        self.ignore_index = ignore_index

        self.image_dir = self.voc_root / "JPEGImages"
        self.mask_dir = self.voc_root / "SegmentationClass"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"VOC directories not found under {self.voc_root}. "
                "Expected JPEGImages and SegmentationClass."
            )

        self.ids = self._load_ids(self.split_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        self.set_attrs(total_len=len(self.ids))

    @staticmethod
    def _load_ids(split_file: Path) -> List[str]:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with split_file.open("r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        if not ids:
            raise RuntimeError(f"Split file is empty: {split_file}")
        return ids

    @staticmethod
    def _resize_short_side(image: Image.Image, mask: Image.Image, short_size: int) -> Tuple[Image.Image, Image.Image]:
        width, height = image.size
        if min(width, height) == short_size:
            return image, mask

        if width <= height:
            out_w = short_size
            out_h = int(round(height * short_size / width))
        else:
            out_h = short_size
            out_w = int(round(width * short_size / height))

        image = image.resize((out_w, out_h), Image.BILINEAR)
        mask = mask.resize((out_w, out_h), Image.NEAREST)
        return image, mask

    @staticmethod
    def _random_crop(
        image: Image.Image,
        mask: Image.Image,
        crop_size: int,
        ignore_index: int,
    ) -> Tuple[Image.Image, Image.Image]:
        width, height = image.size
        pad_w = max(crop_size - width, 0)
        pad_h = max(crop_size - height, 0)

        if pad_w > 0 or pad_h > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=ignore_index)

        width, height = image.size
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)

        image = image.crop((left, top, left + crop_size, top + crop_size))
        mask = mask.crop((left, top, left + crop_size, top + crop_size))
        return image, mask

    @staticmethod
    def _center_crop(
        image: Image.Image,
        mask: Image.Image,
        crop_size: int,
        ignore_index: int,
    ) -> Tuple[Image.Image, Image.Image]:
        width, height = image.size
        pad_w = max(crop_size - width, 0)
        pad_h = max(crop_size - height, 0)

        if pad_w > 0 or pad_h > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=ignore_index)

        width, height = image.size
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        image = image.crop((left, top, left + crop_size, top + crop_size))
        mask = mask.crop((left, top, left + crop_size, top + crop_size))
        return image, mask

    def _train_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        scale = random.uniform(0.5, 2.0)
        short_size = max(32, int(round(self.base_size * scale)))
        image, mask = self._resize_short_side(image, mask, short_size)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image, mask = self._random_crop(image, mask, self.crop_size, self.ignore_index)
        return image, mask

    def _eval_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image, mask = self._resize_short_side(image, mask, self.base_size)
        image, mask = self._center_crop(image, mask, self.crop_size, self.ignore_index)
        return image, mask

    def _image_to_tensor(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self.mean) / self.std
        return arr

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> np.ndarray:
        return np.array(mask, dtype=np.int32)

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        image_path = self.image_dir / f"{sample_id}.jpg"
        mask_path = self.mask_dir / f"{sample_id}.png"

        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing VOC sample files for id={sample_id}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.is_train and self.use_augment:
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = self._eval_transform(image, mask)

        image_arr = self._image_to_tensor(image)
        mask_arr = self._mask_to_tensor(mask)
        return image_arr, mask_arr

    def __len__(self) -> int:
        return len(self.ids)
