import jittor as jt
import numpy as np
from pathlib import Path
from model import UNet
from prepare_voc import prepare_voc_dataset
from train import build_loader, _compute_segmentation_loss
from metrics import RunningScore

jt.flags.use_cuda = 1 if jt.has_cuda else 0

def debug_model():
    print("Loading data...")
    voc_root, split_root = prepare_voc_dataset(
        data_root="./data", val_ratio=0.5, seed=42, force_split=False
    )

    train_loader = build_loader(
        voc_root=voc_root, split_file=split_root / "train.txt",
        is_train=True, use_augment=False, batch_size=4,
        crop_size=320, base_size=512, ignore_index=255, num_workers=0
    )

    model = UNet(in_channels=3, num_classes=21, base_channels=32, use_se=True)
    # 尝试加载可能存在的最好模型
    model_path = Path("runs/unet_se/best_model.pkl")
    if model_path.exists():
        model.load(str(model_path))
        print(f"Loaded {model_path}")
    else:
        print("No pre-trained model found, using random init.")

    score = RunningScore(num_classes=21)

    from train import _compute_class_weights_from_masks
    weights = _compute_class_weights_from_masks(voc_root, split_root / "train.txt", 21, 255, "median_freq", 0.1, 10.0)
    print(f"Calculated Weights: {[round(w, 4) for w in weights]}")

    print("\n--- Testing 3 Batches from Train Loader ---")
    model.train()  # 看看前向特征
    optimizer = jt.nn.AdamW(model.parameters(), lr=0.0005)

    for i, (images, masks) in enumerate(train_loader):
        if i >= 3:
            break
        
        logits = model(images)
        preds = jt.argmax(logits, dim=1)[0]
        
        mask_np = masks.numpy()
        pred_np = preds.numpy()
        
        # Unique classes
        unique_masks = np.unique(mask_np)
        unique_preds = np.unique(pred_np)
        
        print(f"Batch {i}:")
        print(f"  Mask unique classes: {unique_masks}")
        print(f"  Pred unique classes: {unique_preds}")

        loss_ce = _compute_segmentation_loss(
            logits, masks, ignore_index=255, num_classes=21, nn=jt.nn, jt=jt,
            loss_type="ce", dice_weight=0.0
        )
        loss_focal = _compute_segmentation_loss(
            logits, masks, ignore_index=255, num_classes=21, nn=jt.nn, jt=jt,
            loss_type="focal", dice_weight=0.0
        )
        loss_dice = _compute_segmentation_loss(
            logits, masks, ignore_index=255, num_classes=21, nn=jt.nn, jt=jt,
            loss_type="ce", dice_weight=1.0 # (dice_only conceptually if CE is 0, but here ce+dice)
        )
        
        print(f"  CE Loss: {loss_ce.item():.4f}, Focal Loss: {loss_focal.item():.4f}, CE+Dice: {loss_dice.item():.4f}")
        
        # Check gradients
        optimizer.zero_grad()
        optimizer.backward(loss_focal)
        
        grad_norms = []
        for p in model.parameters():
            if p.opt_grad(optimizer) is not None:
                grad_norms.append(np.linalg.norm(p.opt_grad(optimizer).numpy()))
        
        if grad_norms:
            print(f"  Gradient norm stats: min={np.min(grad_norms):.6f}, max={np.max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")

        score.update(pred_np, mask_np)

    scores = score.get_scores()
    print("\n--- Accumulated Metrics ---")
    print(f"  mIoU: {scores['mIoU']:.4f}")
    iou_str = ", ".join([f"{iou:.4f}" for iou in scores['class_iou']])
    print(f"  Class IoU: {iou_str}")

if __name__ == "__main__":
    debug_model()
