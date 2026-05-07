"""Hyperparameter experiments for VOC2012 segmentation training."""

import argparse
import csv
import itertools
import json
from pathlib import Path

from train import get_arg_parser, run_training


def parse_csv_floats(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_bools(text: str):
    values = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "t", "yes", "y"}:
            values.append(True)
        elif token in {"0", "false", "f", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {item}")
    return values


def get_arg_parser_for_tune() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for U-Net + VOC2012")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-root", type=str, default="./runs/tuning")

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.5)

    parser.add_argument("--lrs", type=str, default="0.01,0.003")
    parser.add_argument("--batch-sizes", type=str, default="4,8")
    parser.add_argument("--use-se-options", type=str, default="1,0")

    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr-scheduler", type=str, default="poly", choices=["constant", "poly", "cosine"])
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=3)

    parser.add_argument("--loss-type", type=str, default="ce_dice", choices=["ce", "ce_dice"])
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--dice-eps", type=float, default=1e-6)

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def main() -> None:
    args = get_arg_parser_for_tune().parse_args()

    lrs = parse_csv_floats(args.lrs)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    se_options = parse_csv_bools(args.use_se_options)

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    base_train_args = get_arg_parser().parse_args([])

    results = []

    for lr, batch_size, use_se in itertools.product(lrs, batch_sizes, se_options):
        exp_name = f"lr{lr}_bs{batch_size}_se{int(use_se)}"
        exp_dir = save_root / exp_name

        train_args = argparse.Namespace(**vars(base_train_args))
        train_args.data_root = args.data_root
        train_args.save_dir = str(exp_dir)

        train_args.epochs = args.epochs
        train_args.batch_size = batch_size
        train_args.lr = lr
        train_args.lr_scheduler = args.lr_scheduler
        train_args.min_lr = args.min_lr
        train_args.poly_power = args.poly_power
        train_args.warmup_epochs = args.warmup_epochs

        train_args.loss_type = args.loss_type
        train_args.ce_weight = args.ce_weight
        train_args.dice_weight = args.dice_weight
        train_args.dice_eps = args.dice_eps

        train_args.momentum = args.momentum
        train_args.weight_decay = args.weight_decay

        train_args.crop_size = args.crop_size
        train_args.base_size = args.base_size
        train_args.base_channels = args.base_channels
        train_args.num_workers = args.num_workers

        train_args.use_se = use_se
        train_args.seed = args.seed
        train_args.val_ratio = args.val_ratio
        train_args.force_split = False
        train_args.log_interval = args.log_interval

        print("=" * 80)
        print(f"[tune] run experiment: {exp_name}")
        print("=" * 80)

        result = run_training(train_args)
        results.append(
            {
                "experiment": exp_name,
                "lr": lr,
                "batch_size": batch_size,
                "use_se": use_se,
                "best_epoch": result["best_epoch"],
                "best_val_mIoU": result["best_val_mIoU"],
                "test_loss": result["test_metrics"]["loss"],
                "test_pixel_acc": result["test_metrics"]["pixel_acc"],
                "test_mIoU": result["test_metrics"]["mIoU"],
            }
        )

    results = sorted(results, key=lambda x: x["test_mIoU"], reverse=True)

    csv_path = save_root / "tuning_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = save_root / "tuning_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(8, len(results) * 1.2), 4.5))
    plt.bar([r["experiment"] for r in results], [r["test_mIoU"] for r in results])
    plt.ylabel("test mIoU")
    plt.title("Hyperparameter Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(str(save_root / "tuning_results.png"), dpi=220)
    plt.close()

    print("\n[tune] ranking by test_mIoU")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank:02d}. {item['experiment']:<24} "
            f"test_mIoU={item['test_mIoU']:.4f} "
            f"test_pixel_acc={item['test_pixel_acc']:.4f}"
        )

    print(f"\n[tune] saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
