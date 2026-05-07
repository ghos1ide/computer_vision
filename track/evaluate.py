"""Standalone evaluation for a trained VOC2012 segmentation checkpoint."""

import argparse

from prepare_voc import prepare_voc_dataset
from runtime import import_jittor_or_exit, print_startup_hint
from train import build_loader, run_epoch, set_seed


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint on VOC split")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pkl checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--base-size", type=int, default=512)

    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--base-channels", type=int, default=32)

    parser.add_argument("--use-se", dest="use_se", action="store_true")
    parser.add_argument("--no-se", dest="use_se", action="store_false")
    parser.set_defaults(use_se=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser


def main() -> None:
    args = get_arg_parser().parse_args()

    print_startup_hint("evaluation")
    jt, nn = import_jittor_or_exit("evaluation")

    set_seed(args.seed, jt=jt)
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print(f"[env] jt.has_cuda={jt.has_cuda}, jt.flags.use_cuda={jt.flags.use_cuda}")

    from model import UNet

    voc_root, split_root = prepare_voc_dataset(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_split=False,
    )

    loader = build_loader(
        voc_root=voc_root,
        split_file=split_root / f"{args.split}.txt",
        is_train=False,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        base_size=args.base_size,
        ignore_index=args.ignore_index,
        num_workers=args.num_workers,
    )

    model = UNet(
        in_channels=3,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_se=args.use_se,
    )
    model.load(args.checkpoint)

    metrics = run_epoch(
        model=model,
        loader=loader,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        jt=jt,
        nn=nn,
        optimizer=None,
        log_interval=args.log_interval,
        epoch=0,
        split_name=args.split,
    )

    print("[eval] done")
    print(
        f"[eval] split={args.split} loss={metrics['loss']:.4f} "
        f"pixel_acc={metrics['pixel_acc']:.4f} mIoU={metrics['mIoU']:.4f}"
    )


if __name__ == "__main__":
    main()
