"""Ablation experiments for VOC2012 segmentation."""

import argparse
import csv
import itertools
import json
from pathlib import Path

from train import get_arg_parser, run_training


def get_arg_parser_for_compare() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-root", type=str, default="./runs/ablation")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00005)
    return parser

def main():
    args = get_arg_parser_for_compare().parse_args()
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    
    # define ablation settings
    configs = [
        {"name": "base", "use_se": False, "loss_type": "ce", "dice_weight": 0.0},
        {"name": "se_only", "use_se": True, "loss_type": "ce", "dice_weight": 0.0},
        {"name": "dice_only", "use_se": False, "loss_type": "ce", "dice_weight": 1.0},
        {"name": "focal_only", "use_se": False, "loss_type": "focal", "dice_weight": 0.0},
        {"name": "all", "use_se": True, "loss_type": "focal", "dice_weight": 1.0},
    ]
    
    base_train_args = get_arg_parser().parse_args([])
    results = []
    
    # To store curves data for all configs
    all_histories = {}

    for cfg in configs:
        exp_name = cfg["name"]
        exp_dir = save_root / exp_name

        train_args = argparse.Namespace(**vars(base_train_args))
        train_args.data_root = args.data_root
        train_args.save_dir = str(exp_dir)
        train_args.epochs = args.epochs
        train_args.batch_size = args.batch_size
        train_args.lr = args.lr
        
        train_args.use_se = cfg["use_se"]
        train_args.loss_type = cfg["loss_type"]
        train_args.dice_weight = cfg["dice_weight"]

        print("=" * 80)
        print(f"[ablation] run experiment: {exp_name}")
        print("=" * 80)

        result = run_training(train_args)
        
        # Load history for this config
        history_path = exp_dir / "history.json"
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as f:
                hist_data = json.load(f)
                all_histories[exp_name] = hist_data.get("history", {})

        results.append(
            {
                "experiment": exp_name,
                "best_epoch": result["best_epoch"],
                "best_val_mIoU": result["best_val_mIoU"],
                "test_loss": result["test_metrics"]["loss"],
                "test_pixel_acc": result["test_metrics"]["pixel_acc"],
                "test_mIoU": result["test_metrics"]["mIoU"],
            }
        )

    results = sorted(results, key=lambda x: x["test_mIoU"], reverse=True)

    csv_path = save_root / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = save_root / "ablation_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot overlapping training curves
    if all_histories:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 4))
        
        # Plot Loss
        plt.subplot(1, 3, 1)
        for exp_name, hist in all_histories.items():
            if "val_loss" in hist:
                epochs = range(1, len(hist["val_loss"]) + 1)
                plt.plot(epochs, hist["val_loss"], label=exp_name)
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot Pixel Accuracy
        plt.subplot(1, 3, 2)
        for exp_name, hist in all_histories.items():
            if "val_pixel_acc" in hist:
                epochs = range(1, len(hist["val_pixel_acc"]) + 1)
                plt.plot(epochs, hist["val_pixel_acc"], label=exp_name)
        plt.title("Validation Pixel Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Plot mIoU
        plt.subplot(1, 3, 3)
        for exp_name, hist in all_histories.items():
            if "val_mIoU" in hist:
                epochs = range(1, len(hist["val_mIoU"]) + 1)
                plt.plot(epochs, hist["val_mIoU"], label=exp_name)
        plt.title("Validation mIoU")
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(save_root / "comparison_curves.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
