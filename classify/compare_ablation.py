"""对比随机裁剪、批归一化、Dropout、动量和权重衰减的消融实验结果。"""

from dataclasses import dataclass
import argparse
import random
from typing import List
from PIL import ImageOps

import jittor as jt
from jittor import nn
from jittor.dataset.cifar import CIFAR10
from jittor import transform
import matplotlib.pyplot as plt
import numpy as np

from model import SimpleResNet
from train import train, test


jt.flags.use_cuda = jt.has_cuda

NORMALIZE_MEAN = [0.4914, 0.4822, 0.4465]
NORMALIZE_STD = [0.2470, 0.2435, 0.2616]


@dataclass
class ExperimentConfig:
    name: str
    use_random_crop: bool
    use_batchnorm: bool
    use_dropout: bool
    momentum: float
    weight_decay: float


@dataclass
class ExperimentResult:
    name: str
    final_train_loss: float
    final_test_acc: float
    best_test_acc: float
    train_losses: List[float]
    test_accs: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def build_train_transform(use_random_crop: bool):
    transforms = []
    if use_random_crop:
        transforms.append(transform.Lambda(lambda img: ImageOps.expand(img, border=4, fill=0)))
        transforms.append(transform.RandomCrop(32))
    transforms.append(transform.RandomHorizontalFlip())
    transforms.append(transform.ImageNormalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transform.Compose(transforms)


def build_test_transform():
    return transform.Compose([
        transform.ImageNormalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def build_loaders(batch_size: int, use_random_crop: bool):
    train_loader = CIFAR10(train=True, transform=build_train_transform(use_random_crop)).set_attrs(batch_size=batch_size, shuffle=True)
    test_loader = CIFAR10(train=False, transform=build_test_transform()).set_attrs(batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_model(use_batchnorm: bool, use_dropout: bool):
    return SimpleResNet(
        num_classes=10,
        dropout_prob=0.3,
        use_batchnorm=use_batchnorm,
        use_dropout=use_dropout,
    )


def build_optimizer(model, learning_rate: float, momentum: float, weight_decay: float):
    return nn.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def run_experiment(config: ExperimentConfig, learning_rate: float, batch_size: int, epochs: int, seed: int) -> ExperimentResult:
    set_seed(seed)
    train_loader, test_loader = build_loaders(batch_size, config.use_random_crop)
    model = build_model(config.use_batchnorm, config.use_dropout)
    optimizer = build_optimizer(model, learning_rate, config.momentum, config.weight_decay)

    train_losses: List[float] = []
    test_accs: List[float] = []

    print(f"\n========== {config.name} ==========")
    print(
        f"随机裁剪={config.use_random_crop}, 批归一化={config.use_batchnorm}, Dropout={config.use_dropout}, "
        f"动量={config.momentum}, 权重衰减={config.weight_decay}"
    )
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, train_losses)
        test(model, test_loader, test_accs)

    return ExperimentResult(
        name=config.name,
        final_train_loss=train_losses[-1],
        final_test_acc=test_accs[-1],
        best_test_acc=max(test_accs),
        train_losses=train_losses,
        test_accs=test_accs,
    )


def print_summary(results: List[ExperimentResult]) -> None:
    baseline = results[0]
    print("\n================ 消融实验汇总 ================")
    print("配置\t\t\t最终Acc\t最佳Acc\t最终TrainLoss\t最佳Acc变化\t最终TrainLoss变化")
    for result in results:
        delta_best = result.best_test_acc - baseline.best_test_acc
        delta_loss = result.final_train_loss - baseline.final_train_loss
        print(
            f"{result.name:<14}\t"
            f"{result.final_test_acc * 100:6.2f}%\t"
            f"{result.best_test_acc * 100:6.2f}%\t"
            f"{result.final_train_loss:8.4f}\t"
            f"{delta_best * 100:+7.2f}pp\t"
            f"{delta_loss:+9.4f}"
        )


def plot_curves(results: List[ExperimentResult], output_prefix: str) -> None:
    epochs = range(1, len(results[0].train_losses) + 1)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(epochs, result.train_losses, label=result.name)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(fontsize="small")

    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(epochs, result.test_accs, label=result.name)
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary(results: List[ExperimentResult], output_prefix: str) -> None:
    labels = [result.name for result in results]
    final_accs = [result.final_test_acc * 100 for result in results]
    best_accs = [result.best_test_acc * 100 for result in results]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(16, 6))
    plt.bar(x - width / 2, final_accs, width, label="Final Accuracy")
    plt.bar(x + width / 2, best_accs, width, label="Best Accuracy")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Ablation Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="逐个对比随机裁剪、批归一化、Dropout、动量和权重衰减的影响")
    parser.add_argument("--epochs", type=int, default=20, help="每个配置训练的轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-prefix", type=str, default="ablation_compare", help="输出文件名前缀")
    return parser.parse_args()


def main():
    args = parse_args()

    experiments = [
        ExperimentConfig("基线", True, True, True, 0.9, 1e-4),
        ExperimentConfig("去除随机裁剪", False, True, True, 0.9, 1e-4),
        ExperimentConfig("去除批归一化", True, False, True, 0.9, 1e-4),
        ExperimentConfig("去除Dropout", True, True, False, 0.9, 1e-4),
        ExperimentConfig("去除动量", True, True, True, 0.0, 1e-4),
        ExperimentConfig("去除权重衰减", True, True, True, 0.9, 0.0),
    ]

    results = []
    for index, config in enumerate(experiments):
        result = run_experiment(
            config=config,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )
        results.append(result)

    print_summary(results)
    plot_curves(results, args.output_prefix)
    plot_summary(results, args.output_prefix)
    print(f"\n结果图已保存为 {args.output_prefix}_curves.png 和 {args.output_prefix}_summary.png")


if __name__ == "__main__":
    main()