"""Segmentation metrics for pixel accuracy and IoU."""

import numpy as np


class RunningScore:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray) -> np.ndarray:
        valid = label_true != self.ignore_index
        label_true = label_true[valid].astype(np.int64)
        label_pred = label_pred[valid].astype(np.int64)

        mask = (label_true >= 0) & (label_true < self.num_classes)
        label_true = label_true[mask]
        label_pred = label_pred[mask]

        hist = np.bincount(
            self.num_classes * label_true + label_pred,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_preds: np.ndarray, label_trues: np.ndarray) -> None:
        if label_preds.ndim == 2:
            label_preds = label_preds[None, ...]
            label_trues = label_trues[None, ...]

        for lp, lt in zip(label_preds, label_trues):
            self.confusion_matrix += self._fast_hist(lt, lp)

    def get_scores(self):
        hist = self.confusion_matrix
        eps = 1e-10

        pixel_acc = np.diag(hist).sum() / (hist.sum() + eps)

        class_acc = np.diag(hist) / (hist.sum(axis=1) + eps)
        mean_acc = np.nanmean(class_acc)

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + eps)
        mean_iu = np.nanmean(iu)

        freq = hist.sum(axis=1) / (hist.sum() + eps)
        fw_iu = (freq * iu).sum()

        return {
            "pixel_acc": float(pixel_acc),
            "mean_acc": float(mean_acc),
            "mIoU": float(mean_iu),
            "fwIoU": float(fw_iu),
            "class_iou": iu.tolist(),
        }
