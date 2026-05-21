# test.py
# purpose: evaluate a trained multi-abnormality classifier checkpoint on the held-out test split
# author: Anthony Smaldore

import os
import sys
import json
import argparse
import logging

import torch
from monai.data import Dataset, DataLoader
from pytorch_lightning import Trainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.transforms import get_valid_transforms
from modules.multiabnormality_classification_model import MultiAbnormalityClassifier

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments to set the training parameters and output specifications
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate Multi-Abnormality Classifier')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the .ckpt file to evaluate')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to patient data directory (same as train.py)')
    parser.add_argument('--test_split', type=str, default=None,
                        help='Path to test_split.json (defaults to <data_dir>/subset_manifest/test_split.json)')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--spatial_size', type=int, nargs=3, default=(96, 224, 224))
    parser.add_argument('--pixdim', type=float, nargs=3, default=(1.5, 1.5, 1.5))

    parser.add_argument('--use_kfold', action='store_true', help='Evaluate a specific fold from the k-fold cross validation')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to evaluate')

    return parser.parse_args()


def build_test_loader(test_data, args):
    """
    Build the test dataloader
    Args:
        test_data: Test data
        args: Parsed command line arguments
    Returns:
        test_loader: Test dataloader
    """
    transforms = get_valid_transforms(
        spatial_size=tuple(args.spatial_size),
        pixdim=tuple(args.pixdim),
    )
    ds = Dataset(data=test_data, transform=transforms)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    args = parse_args()

    if args.test_split:
        test_split_path = args.test_split
    elif getattr(args, 'use_kfold', False):
        test_split_path = os.path.join(
            args.data_dir, 'subset_manifest', f'fold_{args.fold}', 'val_split.json'
        )
    else:
        test_split_path = os.path.join(
            args.data_dir, 'subset_manifest', 'test_split.json'
        )

    if not os.path.exists(test_split_path):
        raise FileNotFoundError(
            f"test_split.json not found at {test_split_path}. "
            f"Run train.py first (or pass --test_split)."
        )
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logger.info(f"Loading test split from {test_split_path}")
    with open(test_split_path, 'r') as f:
        test_data = json.load(f)
    logger.info(f"Test samples: {len(test_data)}")

    logger.info("Building test dataloader...")
    test_loader = build_test_loader(test_data, args)

    logger.info(f"Loading model from {args.checkpoint}")
    model = MultiAbnormalityClassifier.load_from_checkpoint(args.checkpoint)
    model.eval()

    logger.info(f"Restored thresholds: {model.thresholds.cpu().tolist()}")

    trainer = Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )
    results = trainer.test(model, test_loader)
    logger.info(f"Test results: {results}")


if __name__ == '__main__':
    main()