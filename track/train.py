"""Training script for VOC2012 semantic segmentation with Jittor U-Net."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

def set_seed(seed: int, jt=None) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if jt is not None:
        jt.set_global_seed(seed)


def _to_numpy(x):
    import numpy as np

    if hasattr(x, "data"):
        return x.data
    return np.array(x)


def _load_split_ids(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with split_file.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Split file is empty: {split_file}")
    return ids


def _compute_class_weights_from_masks(
    voc_root: Path,
    split_file: Path,
    num_classes: int,
    ignore_index: int,
    method: str,
    min_weight: float,
    max_weight: float,
) -> List[float]:
    from PIL import Image
    import numpy as np

    mask_dir = voc_root / "SegmentationClass"
    ids = _load_split_ids(split_file)

    counts = np.zeros(num_classes, dtype=np.float64)
    for sample_id in ids:
        mask_path = mask_dir / f"{sample_id}.png"
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        valid = mask != ignore_index
        if not np.any(valid):
            continue

        hist = np.bincount(mask[valid].ravel(), minlength=num_classes)
        counts += hist[:num_classes]

    total = float(counts.sum())
    if total <= 0:
        return [1.0 for _ in range(num_classes)]

    freq = counts / total
    weights = np.zeros_like(freq)
    present = freq > 0

    if method == "median_freq":
        median = float(np.median(freq[present])) if np.any(present) else 1.0
        weights[present] = median / freq[present]
    elif method == "sqrt_inv":
        weights[present] = 1.0 / np.sqrt(freq[present])
    elif method == "log_inv":
        weights[present] = 1.0 / np.log(1.02 + freq[present])
    else:
        weights[present] = 1.0

    weights = np.clip(weights, min_weight, max_weight)
    return weights.astype(np.float32).tolist()


def _compute_dice_loss_dense(
    probs,
    masks,
    ignore_index: int,
    jt,
    eps: float,
    class_weights=None,
    include_background: bool = False,
):
    masks_valid = (masks != ignore_index)

    masks_clean = masks.clone()
    masks_clean[masks == ignore_index] = 0
    masks_clean = masks_clean.unsqueeze(1)

    targets_one_hot = jt.zeros_like(probs)
    targets_one_hot.scatter_(1, masks_clean, jt.ones_like(masks_clean))

    intersection = (probs * targets_one_hot * masks_valid.unsqueeze(1)).sum(dims=(2, 3))
    cardinality = ((probs + targets_one_hot) * masks_valid.unsqueeze(1)).sum(dims=(2, 3))
    dice_loss = 1.0 - (2.0 * intersection + eps) / (cardinality + eps)

    if not include_background:
        dice_loss = dice_loss[:, 1:]
        if class_weights is not None:
            class_weights = class_weights[1:]

    if class_weights is None:
        return dice_loss.mean()

    weights = class_weights.reshape(1, -1)
    weight_sum = float(weights.sum().item())
    if weight_sum <= 0:
        return dice_loss.mean()

    weighted = (dice_loss * weights).sum(dims=1) / weight_sum
    return weighted.mean()


def _compute_dice_loss_sparse(
    probs,
    masks,
    num_classes: int,
    ignore_index: int,
    jt,
    eps: float,
    class_weights=None,
    include_background: bool = False,
):
    valid = (masks != ignore_index)
    dice_total = None
    dice_count = 0
    weight_total = 0.0

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        if not include_background and cls == 0:
            continue

        prob_c = probs[:, cls, :, :] * valid
        target_c = (masks == cls)

        intersection = (prob_c * target_c).sum(dims=(1, 2))
        cardinality = prob_c.sum(dims=(1, 2)) + target_c.sum(dims=(1, 2))

        dice = 1.0 - (2.0 * intersection + eps) / (cardinality + eps)
        dice_mean = dice.mean()

        if class_weights is not None:
            weight = class_weights[cls]
            if float(weight.item()) <= 0:
                continue
            dice_mean = dice_mean * weight
            weight_total += float(weight.item())
        else:
            dice_count += 1

        dice_total = dice_mean if dice_total is None else dice_total + dice_mean

    if dice_total is None:
        return probs.sum() * 0.0
    if class_weights is not None:
        return dice_total / max(weight_total, 1e-6)
    return dice_total / max(dice_count, 1)


def _compute_segmentation_loss(
    logits,
    masks,
    ignore_index: int,
    num_classes: int,
    nn,
    jt,
    class_weights=None,
    dice_weight: float = 0.5,
    dice_mode: str = "sparse",
    dice_eps: float = 1e-5,
    dice_include_background: bool = False,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
):
    valid_mask = (masks != ignore_index)
    
    if loss_type == "ce":
        if class_weights is None:
            ce_loss = nn.cross_entropy_loss(logits, masks, ignore_index=ignore_index)
        else:
            # 引入类 MMSegmentation 的加权 CE，手动降维避免框架 API 兼容问题
            log_probs = nn.log_softmax(logits, dim=1)
            masks_safe = jt.clamp(masks, 0, num_classes - 1)
            masks_1hot = jt.zeros_like(logits).scatter_(1, masks_safe.unsqueeze(1), jt.ones_like(masks_safe.unsqueeze(1)))
            ce_loss_unreduced = -(log_probs * masks_1hot).sum(dim=1)
            
            cw_tensor = jt.array(class_weights)
            cw_map = cw_tensor[masks_safe]
            ce_loss_unreduced = ce_loss_unreduced * cw_map
            
            valid_losses = ce_loss_unreduced[valid_mask]
            weight_sum = cw_map[valid_mask].sum()
            ce_loss = valid_losses.sum() / jt.maximum(weight_sum, 1e-6)

    elif loss_type == "focal":
        # 引入 MMDetection / MMSegmentation 经典的 Focal Loss
        log_probs = nn.log_softmax(logits, dim=1)
        masks_safe = jt.clamp(masks, 0, num_classes - 1)
        masks_1hot = jt.zeros_like(logits).scatter_(1, masks_safe.unsqueeze(1), jt.ones_like(masks_safe.unsqueeze(1)))
        
        ce_loss_unreduced = -(log_probs * masks_1hot).sum(dim=1)
        pt_probs = jt.exp(-ce_loss_unreduced)
        focal_weight_map = (1.0 - pt_probs) ** focal_gamma
        
        loss_unreduced = ce_loss_unreduced * focal_weight_map
        if class_weights is not None:
            cw_tensor = jt.array(class_weights)
            cw_map = cw_tensor[masks_safe]
            loss_unreduced = loss_unreduced * cw_map
            
        valid_losses = loss_unreduced[valid_mask]
        
        if class_weights is not None:
            weight_sum = cw_map[valid_mask].sum()
            ce_loss = valid_losses.sum() / jt.maximum(weight_sum, 1e-6)
        else:
            ce_loss = valid_losses.mean() if valid_losses.numel() > 0 else loss_unreduced.sum() * 0.0
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if dice_weight <= 0:
        return ce_loss

    probs = nn.softmax(logits, dim=1)
    if dice_mode == "dense":
        dice_loss = _compute_dice_loss_dense(
            probs,
            masks,
            ignore_index,
            jt,
            dice_eps,
            class_weights=class_weights,
            include_background=dice_include_background,
        )
    elif dice_mode == "sparse":
        dice_loss = _compute_dice_loss_sparse(
            probs,
            masks,
            num_classes,
            ignore_index,
            jt,
            dice_eps,
            class_weights=class_weights,
            include_background=dice_include_background,
        )
    else:
        raise ValueError(f"Unknown dice_mode: {dice_mode}")

    return ce_loss + dice_weight * dice_loss


def _set_optimizer_lr(optimizer, lr: float) -> None:
    optimizer.lr = lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _estimate_steps_per_epoch(loader) -> int:
    total_len = getattr(loader, "total_len", None)
    batch_size = getattr(loader, "batch_size", None)
    drop_last = bool(getattr(loader, "drop_last", False))

    if total_len is not None and batch_size is not None:
        total_len = int(total_len)
        batch_size = int(batch_size)
        if batch_size > 0:
            if drop_last:
                return max(total_len // batch_size, 1)
            return max((total_len + batch_size - 1) // batch_size, 1)

    try:
        return max(int(len(loader)), 1)
    except Exception:
        return 1


def _compute_learning_rate(
    scheduler: str,
    base_lr: float,
    max_lr: float,
    min_lr: float,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps - 1), 0)

    if scheduler == "constant":
        return base_lr

    if scheduler == "cosine":
        if warmup_steps > 0 and global_step < warmup_steps:
            alpha = float(global_step + 1) / float(warmup_steps)
            return base_lr + (max_lr - base_lr) * alpha

        after_total = max(total_steps - warmup_steps, 1)
        after_step = max(global_step - warmup_steps, 0)
        progress = min(max(float(after_step) / float(after_total), 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    if scheduler == "onecycle":
        up_steps = warmup_steps if warmup_steps > 0 else max(1, int(total_steps * 0.3))
        up_steps = max(min(up_steps, total_steps - 1), 1)
        if global_step < up_steps:
            alpha = float(global_step + 1) / float(up_steps)
            return base_lr + (max_lr - base_lr) * alpha

        down_total = max(total_steps - up_steps, 1)
        down_step = max(global_step - up_steps, 0)
        progress = min(max(float(down_step) / float(down_total), 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    raise ValueError(f"Unknown lr scheduler: {scheduler}")


def build_loader(
    voc_root: Path,
    split_file: Path,
    is_train: bool,
    use_augment: bool,
    batch_size: int,
    crop_size: int,
    base_size: int,
    ignore_index: int,
    num_workers: int,
):
    from dataset import VOCSegDataset

    dataset = VOCSegDataset(
        voc_root=voc_root,
        split_file=split_file,
        is_train=is_train,
        use_augment=use_augment,
        crop_size=crop_size,
        base_size=base_size,
        ignore_index=ignore_index,
    )
    dataset.set_attrs(
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
    )
    return dataset


def run_epoch(
    model,
    loader,
    num_classes: int,
    ignore_index: int,
    jt,
    nn,
    class_weights=None,
    dice_weight: float = 0.5,
    dice_mode: str = "sparse",
    dice_eps: float = 1e-5,
    dice_include_background: bool = False,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    lr_scheduler_cfg=None,
    optimizer=None,
    log_interval: int = 20,
    epoch: int = 0,
    split_name: str = "train",
    steps_per_epoch: int = 0,
):
    from metrics import RunningScore

    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    score = RunningScore(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    total_batches = 0
    total_steps = max(int(steps_per_epoch), 1) if steps_per_epoch else _estimate_steps_per_epoch(loader)
    last_lr = None

    def _loop_body():
        nonlocal total_loss, total_batches, last_lr
        for batch_idx, (images, masks) in enumerate(loader):
            if is_train and lr_scheduler_cfg is not None:
                current_lr = _compute_learning_rate(
                    scheduler=lr_scheduler_cfg["scheduler"],
                    base_lr=lr_scheduler_cfg["base_lr"],
                    max_lr=lr_scheduler_cfg["max_lr"],
                    min_lr=lr_scheduler_cfg["min_lr"],
                    global_step=lr_scheduler_cfg["global_step"],
                    total_steps=lr_scheduler_cfg["total_steps"],
                    warmup_steps=lr_scheduler_cfg["warmup_steps"],
                )
                _set_optimizer_lr(optimizer, current_lr)
                lr_scheduler_cfg["global_step"] += 1
                last_lr = current_lr

            logits = model(images)
            loss = _compute_segmentation_loss(
                logits=logits,
                masks=masks,
                ignore_index=ignore_index,
                num_classes=num_classes,
                nn=nn,
                jt=jt,
                class_weights=class_weights,
                dice_weight=dice_weight,
                dice_mode=dice_mode,
                dice_eps=dice_eps,
                dice_include_background=dice_include_background,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
            )

            if is_train:
                optimizer.step(loss)

            total_loss += float(loss.item())
            total_batches += 1

            logits_np = _to_numpy(logits.detach())
            pred_np = logits_np.argmax(axis=1)
            mask_np = _to_numpy(masks.detach())

            if pred_np.shape != mask_np.shape:
                raise RuntimeError(
                    f"Prediction/label shape mismatch: pred={pred_np.shape}, mask={mask_np.shape}. "
                    "Expected [B,H,W] for both prediction and mask."
                )

            score.update(pred_np, mask_np)

            if batch_idx % max(1, log_interval) == 0:
                lr_info = f" lr={last_lr:.6f}" if last_lr is not None else ""
                print(
                    f"[{split_name}] epoch={epoch:03d} "
                    f"batch={batch_idx:04d}/{total_steps:04d} "
                    f"loss={float(loss.item()):.4f}{lr_info}"
                )

    if is_train:
        _loop_body()
    else:
        with jt.no_grad():
            _loop_body()

    avg_loss = total_loss / max(total_batches, 1)
    scores = score.get_scores()
    
    # ---- 诊断输出，帮助观察是否全预测成了背景 ----
    if split_name == "val" or split_name == "test":
        print(f"\n[Diagnostic - {split_name} epoch {epoch:03d}]")
        print(f"  Pixel Acc: {scores['pixel_acc']:.4f}, Mean Acc: {scores['mean_acc']:.4f}, mIoU: {scores['mIoU']:.4f}")
        
        # 统计验证集中出现的真实类别和预测出的类别分布
        pred_freq = score.confusion_matrix.sum(axis=0)  # 预测统计
        true_freq = score.confusion_matrix.sum(axis=1)  # 真实统计
        
        print("  Class IoU Details:")
        class_ious = scores.get('class_iou', [])
        for cls_idx in range(num_classes):
            if true_freq[cls_idx] > 0 or pred_freq[cls_idx] > 0:
                iou_val = class_ious[cls_idx] if cls_idx < len(class_ious) else 0.0
                print(f"    Class {cls_idx:02d}: IoU = {iou_val:.4f} | Pred count: {pred_freq[cls_idx]} | True count: {true_freq[cls_idx]}")
        print("-----------------------------------------\n")

    return {
        "loss": avg_loss,
        "pixel_acc": scores["pixel_acc"],
        "mIoU": scores["mIoU"],
        "fwIoU": scores["fwIoU"],
        "mean_acc": scores["mean_acc"],
        "lr": float(last_lr) if last_lr is not None else (float(optimizer.lr) if is_train else None),
    }


def plot_history(history: Dict[str, list], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_pixel_acc"], label="train_pixel_acc")
    plt.plot(epochs, history["val_pixel_acc"], label="val_pixel_acc")
    plt.title("Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_mIoU"], label="train_mIoU")
    plt.plot(epochs, history["val_mIoU"], label="val_mIoU")
    plt.title("mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close()


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train U-Net on VOC2012 segmentation")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root for VOC and split files")
    parser.add_argument("--save-dir", type=str, default="./runs/unet_se", help="Output directory")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["constant", "cosine", "onecycle"])
    parser.add_argument("--max-lr", type=float, default=0.002)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)

    parser.add_argument("--loss-type", type=str, default="focal", choices=["ce", "focal"], help="MMSegmentation classic loss choice")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--dice-mode", type=str, default="sparse", choices=["dense", "sparse"])
    parser.add_argument("--dice-eps", type=float, default=1e-5)
    parser.add_argument("--dice-include-background", dest="dice_include_background", action="store_true")
    parser.add_argument("--dice-exclude-background", dest="dice_include_background", action="store_false")
    parser.set_defaults(dice_include_background=False)

    parser.add_argument(
        "--class-weighting",
        type=str,
        default="median_freq",
        choices=["none", "median_freq", "sqrt_inv", "log_inv"],
    )
    parser.add_argument("--class-weight-min", type=float, default=0.1)
    parser.add_argument("--class-weight-max", type=float, default=10.0)

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--use-aug", dest="use_aug", action="store_true")
    parser.add_argument("--no-aug", dest="use_aug", action="store_false")
    parser.set_defaults(use_aug=True)

    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--use-se", dest="use_se", action="store_true")
    parser.add_argument("--no-se", dest="use_se", action="store_false")
    parser.set_defaults(use_se=False)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.5)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def run_training(args: argparse.Namespace):
    from prepare_voc import prepare_voc_dataset
    from runtime import import_jittor_or_exit, print_startup_hint

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print_startup_hint("training")
    jt, nn = import_jittor_or_exit("training")

    set_seed(args.seed, jt=jt)
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print(f"[env] jt.has_cuda={jt.has_cuda}, jt.flags.use_cuda={jt.flags.use_cuda}")

    from model import UNet

    voc_root, split_root = prepare_voc_dataset(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_split=args.force_split,
    )

    train_loader = build_loader(
        voc_root=voc_root,
        split_file=split_root / "train.txt",
        is_train=True,
        use_augment=args.use_aug,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        base_size=args.base_size,
        ignore_index=args.ignore_index,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        voc_root=voc_root,
        split_file=split_root / "val.txt",
        is_train=False,
        use_augment=False,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        base_size=args.base_size,
        ignore_index=args.ignore_index,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        voc_root=voc_root,
        split_file=split_root / "test.txt",
        is_train=False,
        use_augment=False,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        base_size=args.base_size,
        ignore_index=args.ignore_index,
        num_workers=args.num_workers,
    )

    class_weights = None
    if args.class_weighting != "none" and args.dice_weight > 0:
        weights = _compute_class_weights_from_masks(
            voc_root=voc_root,
            split_file=split_root / "train.txt",
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            method=args.class_weighting,
            min_weight=args.class_weight_min,
            max_weight=args.class_weight_max,
        )
        class_weights = jt.array(weights)

    model = UNet(
        in_channels=3,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_se=args.use_se,
    )

    optimizer = nn.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = _estimate_steps_per_epoch(train_loader)
    base_lr = args.lr
    max_lr = max(args.max_lr, base_lr)
    min_lr = min(args.min_lr, base_lr)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    lr_scheduler_cfg = {
        "scheduler": args.lr_scheduler,
        "base_lr": base_lr,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "global_step": 0,
        "total_steps": max(1, args.epochs * steps_per_epoch),
        "warmup_steps": warmup_steps,
    }

    optimizer_name = optimizer.__class__.__name__.lower()
    if args.dice_weight <= 0:
        loss_desc = "ce"
    else:
        loss_desc = f"ce+dice(w={args.dice_weight}, mode={args.dice_mode})"
    class_weight_desc = args.class_weighting if class_weights is not None else "none"

    print(
        f"[recipe] loss={loss_desc} optimizer={optimizer_name} lr={base_lr} max_lr={max_lr} min_lr={min_lr} "
        f"scheduler={args.lr_scheduler} warmup_epochs={args.warmup_epochs} momentum={args.momentum} "
        f"weight_decay={args.weight_decay} class_weighting={class_weight_desc} "
        f"steps_per_epoch={steps_per_epoch}"
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_pixel_acc": [],
        "val_pixel_acc": [],
        "train_mIoU": [],
        "val_mIoU": [],
    }

    best_val_miou = -1.0
    best_epoch = -1
    best_model_path = save_dir / "best_model.pkl"

    print("[train] start training")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            jt=jt,
            nn=nn,
            class_weights=class_weights,
            dice_weight=args.dice_weight,
            dice_mode=args.dice_mode,
            dice_eps=args.dice_eps,
            dice_include_background=args.dice_include_background,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            lr_scheduler_cfg=lr_scheduler_cfg,
            optimizer=optimizer,
            log_interval=args.log_interval,
            epoch=epoch,
            split_name="train",
            steps_per_epoch=steps_per_epoch,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            jt=jt,
            nn=nn,
            class_weights=class_weights,
            dice_weight=args.dice_weight,
            dice_mode=args.dice_mode,
            dice_eps=args.dice_eps,
            dice_include_background=args.dice_include_background,
            loss_type="ce", # validation generally uses CE/raw
            focal_gamma=args.focal_gamma,
            optimizer=None,
            log_interval=args.log_interval,
            epoch=epoch,
            split_name="val",
            steps_per_epoch=0,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_pixel_acc"].append(train_metrics["pixel_acc"])
        history["val_pixel_acc"].append(val_metrics["pixel_acc"])
        history["train_mIoU"].append(train_metrics["mIoU"])
        history["val_mIoU"].append(val_metrics["mIoU"])

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_mIoU={train_metrics['mIoU']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['mIoU']:.4f} "
            f"lr={(train_metrics['lr'] if train_metrics['lr'] is not None else args.lr):.6f}"
        )

        if val_metrics["mIoU"] > best_val_miou:
            best_val_miou = val_metrics["mIoU"]
            best_epoch = epoch
            model.save(str(best_model_path))
            print(f"[checkpoint] saved best model to {best_model_path}")

    if best_model_path.exists():
        model.load(str(best_model_path))

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        jt=jt,
        nn=nn,
        class_weights=class_weights,
        dice_weight=args.dice_weight,
        dice_mode=args.dice_mode,
        dice_eps=args.dice_eps,
        dice_include_background=args.dice_include_background,
        loss_type="ce", # testing generally uses raw evaluation losses
        focal_gamma=args.focal_gamma,
        optimizer=None,
        log_interval=args.log_interval,
        epoch=best_epoch,
        split_name="test",
        steps_per_epoch=0,
    )

    history_path = save_dir / "history.json"
    history_payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mIoU": best_val_miou,
        "test_metrics": test_metrics,
    }
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    plot_history(history, save_dir / "training_curves.png")

    print("[train] finished")
    print(f"[result] best_epoch={best_epoch}, best_val_mIoU={best_val_miou:.4f}")
    print(
        f"[result] test_loss={test_metrics['loss']:.4f}, "
        f"test_pixel_acc={test_metrics['pixel_acc']:.4f}, test_mIoU={test_metrics['mIoU']:.4f}"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_mIoU": best_val_miou,
        "test_metrics": test_metrics,
        "save_dir": str(save_dir),
    }


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
