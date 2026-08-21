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
from modules.transforms import get_valid_transforms, window_set_from_checkpoint
from modules.multiabnormality_classification_model import MultiAbnormalityClassifier

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parses CLI arguments for checkpoint evaluation on the test split.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
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
    parser.add_argument(
        '--window_set',
        type=str,
        default=None,
        help='Override window set. Default: read from checkpoint, else default layout',
    )

    parser.add_argument('--use_kfold', action='store_true', help='Evaluate a specific fold from the k-fold cross validation')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to evaluate')

    return parser.parse_args()


def build_test_loader(test_data, args, window_set):
    """
    Builds an unshuffled test DataLoader with the named window set.
    
    Args:
        test_data (list): Split records with "image" and "label" keys.
        args (argparse.Namespace): CLI arguments (spatial_size, pixdim, batch_size, num_workers).
        window_set (str): The name of the window set ("default" or "lung_mediastinum_bone").
    
    Returns:
        DataLoader: Unshuffled loader over the test split.
    """
    transforms = get_valid_transforms(
        spatial_size=tuple(args.spatial_size),
        pixdim=tuple(args.pixdim),
        window_set=window_set,
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
    """
    Evaluates a trained checkpoint on the held-out test split.
    
    Uses a k-fold validation split when "--use_kfold" is set.
    
    Raises:
        FileNotFoundError: If the split JSON or the checkpoint path does not exist.
    """
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

    window_set = window_set_from_checkpoint(
        args.checkpoint, override=args.window_set
    )
    logger.info(f"Window set: {window_set} (from checkpoint unless --window_set set)")

    logger.info(f"Loading test split from {test_split_path}")
    with open(test_split_path, 'r') as f:
        test_data = json.load(f)
    logger.info(f"Test samples: {len(test_data)}")

    logger.info("Building test dataloader...")
    test_loader = build_test_loader(test_data, args, window_set)

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