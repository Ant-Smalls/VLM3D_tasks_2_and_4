# train.py 
# purpose: train the multi-abnormality classification model using a pretrained encoder and custom classification head
# author: Anthony Smaldore 

import os
import sys
import json
import argparse
import random
import logging
import torch
from monai.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.data_loader import create_subset_manifest
from modules.data_loader import create_train_val_test_splits
from modules.data_loader import create_stratified_kfold_splits
from modules.transforms import (
    get_train_transforms,
    get_valid_transforms,
    KNOWN_WINDOW_SETS,
    resolve_window_set,
)
from modules.multiabnormality_classification_model import MultiAbnormalityClassifier

logger = logging.getLogger(__name__)

def parse_args():
    """
    Parses CLI arguments for training the multi-abnormality classifier.
    
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description='Train Multi-Abnormality Classifier')

    # data paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to patient data directory')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to training labels CSV')
    parser.add_argument('--encoder_dir', type=str, required=True,
                        help='Path to pretrained encoder directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    
    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--encoder_type', type=str, default='dinov2',
                    choices=['dinov2', 'dinov3'],
                    help='Type of encoder to use')
    # LoRA is DINOv2-only; the default path stays fully frozen
    parser.add_argument('--use_lora', action='store_true',
                        help='Attach LoRA adapters on DINOv2 attention (Q/K/V)')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank (only used with --use_lora)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (only used with --use_lora)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (only used with --use_lora)')
    parser.add_argument('--lora_targets', type=str, default='query,key,value',
                        help='Comma-separated peft target module names')
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                        help='Learning rate for LoRA adapter params (only used with --use_lora)')
    # Top-k pooling; default 0 is full softmax and matches frozen-600
    parser.add_argument('--attn_topk', type=int, default=0,
                        help='If >0, keep top-k slice attention weights per class then renormalize')
    parser.add_argument(
        '--window_set',
        type=str,
        default='default',
        choices=list(KNOWN_WINDOW_SETS),
        help='Named 3-channel HU layout (default keeps frozen-600 preprocess)',
    )
    # K-fold settings 
    parser.add_argument('--use_kfold', action='store_true', help='Use multilabel stratified k-fold cross validation')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to train (0 to k_folds-1)')
    
    # additional settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training (rest split equally for val/test)')
    
    return parser.parse_args()



def prepare_data(args):
    """
    Builds or loads the subset manifest and train/val/test splits.
    
    When "--use_kfold" is set, test_data is the fold validation split.
    
    Args:
        args (argparse.Namespace): CLI arguments (data_dir, labels_csv, use_kfold, k_folds, fold, train_split, seed).
    
    Returns:
        tuple: (train_data, val_data, test_data) split records with "image" and "label" keys.
    """
    
    # set directory to contain the patient data manifest 
    manifest_dir = os.path.join(args.data_dir, 'subset_manifest')
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, 'subset_manifest.json')
    
    # if no manifest exists create a new one with the create_subset_manifest function
    if os.path.exists(manifest_path):
        logger.info(f"Using existing manifest at {manifest_path}")
    else:
        logger.info(f"Creating new manifest at {manifest_path}")
        create_subset_manifest(
            master_csv_path=args.labels_csv,
            base_data_dir=args.data_dir,
            output_json_path=manifest_path
        )
    
    if args.use_kfold:
        folds_exist = all([
            os.path.exists(os.path.join(manifest_dir, f'fold_{i}', 'train_split.json'))
            for i in range(args.k_folds)
        ])

        if folds_exist:
            logger.info(f"Using existing {args.k_folds}-fold splits")
        else:
            logger.info(f"Creating new {args.k_folds}-fold splits")
            create_stratified_kfold_splits(
                manifest_path=manifest_path,
                output_dir=manifest_dir,
                k_folds=args.k_folds,
                seed=args.seed
            )
        
        # load specific fold requested by the user 
        train_split_path = os.path.join(manifest_dir, f'fold_{args.fold}', 'train_split.json')
        validation_split_path = os.path.join(manifest_dir, f'fold_{args.fold}', 'val_split.json')

        with open(train_split_path, 'r') as f:
            train_data = json.load(f)
        with open(validation_split_path, 'r') as f:
            val_data = json.load(f)
        
        test_data = val_data
    
    else:
        # set paths for the train/val/test splits and check if they exist
        train_split_path = os.path.join(manifest_dir, 'train_split.json')
        val_split_path = os.path.join(manifest_dir, 'val_split.json')
        test_split_path = os.path.join(manifest_dir, 'test_split.json')
        
        splits_exist = all([
            os.path.exists(train_split_path),
            os.path.exists(val_split_path),
            os.path.exists(test_split_path)
        ])

        # use the existing splits if they exist otherwise create new ones with the create_train_val_test_splits function
        if splits_exist:
            logger.info(f"Using existing splits from {manifest_dir}")
        else:
            logger.info(f"Creating new train/val/test splits...")
            create_train_val_test_splits(
                manifest_path=manifest_path,
                output_dir=manifest_dir,
                train_ratio=args.train_split,
                seed=args.seed
            )
        
        # load the train/val/test splits and return them 
        logger.info("Loading train/val/test data...")
        with open(train_split_path, 'r') as f:
            train_data = json.load(f)
        with open(val_split_path, 'r') as f:
            val_data = json.load(f)
        with open(test_split_path, 'r') as f:
            test_data = json.load(f)
    
    logger.info(f"Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data, args):
    """
    Builds train, validation, and test DataLoaders with the named window set.
    
    Args:
        train_data (list): Training split records with "image" and "label" keys.
        val_data (list): Validation split records with "image" and "label" keys.
        test_data (list): Test split records with "image" and "label" keys.
        args (argparse.Namespace): CLI arguments (batch_size, num_workers, window_set).
    
    Returns:
        tuple: (train_loader, val_loader, test_loader).
    """
    
    # create the training and validation transforms with the get_train_transforms and get_valid_transforms functions
    train_transforms = get_train_transforms(
        spatial_size=(96, 224, 224),
        pixdim=(1.5, 1.5, 1.5),
        window_set=args.window_set,
    )
    
    val_transforms = get_valid_transforms(
        spatial_size=(96, 224, 224),
        pixdim=(1.5, 1.5, 1.5),
        window_set=args.window_set,
    )
    
    # create the training, validation, and test datasets
    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    test_ds = Dataset(data=test_data, transform=val_transforms)

    
    # create the training and validation dataloaders and return 
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def calculate_position_weights(train_dataset, num_classes):
    """
    Computes per-class positive weights for BCEWithLogitsLoss from the training labels.
    
    Args:
        train_dataset (list): Training split records with a "label" key.
        num_classes (int): Number of abnormality classes.
    
    Returns:
        Tensor: Per-class positive weights, clamped to [1.0, 5.0].
    """
    label_counts = torch.zeros(num_classes)
    
    for item in train_dataset:
        labels = item['label']  
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.float32)
        label_counts += labels
    
    num_samples = len(train_dataset)
    num_positives = label_counts
    num_negatives = num_samples - num_positives
    
    # Avoid division by zero
    pos_weight = num_negatives / (num_positives + 1e-5)
    
    # clip extreme weights
    pos_weight = torch.clamp(pos_weight, min=1.0, max=5.0)
    
    return pos_weight

def main():
    """
    Trains the multi-abnormality classifier and evaluates the best checkpoint on the test split.
    
    Raises:
        ValueError: If LoRA is requested with a non-"dinov2" encoder, LoRA targets are empty, "--attn_topk" is negative, or the window-set name is unknown.
    """
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # parse the command line arguments and set the random seed
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.use_lora and args.encoder_type != 'dinov2':
        raise ValueError(
            f"--use_lora requires --encoder_type dinov2 (got {args.encoder_type!r})"
        )
    lora_targets = [t.strip() for t in args.lora_targets.split(',') if t.strip()]
    if args.use_lora and not lora_targets:
        raise ValueError("--lora_targets must list at least one module name when --use_lora is set")
    if args.attn_topk < 0:
        raise ValueError(f"--attn_topk must be >= 0 (got {args.attn_topk})")
    resolve_window_set(args.window_set)
    
    # prepare the data and create the dataloaders
    logger.info("Preparing data...")
    train_data, val_data, test_data = prepare_data(args)
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, args)
    
    # initialize the model with the MultiAbnormalityClassifier class and position weights for loss calculation
    logger.info("Initializing model...")
    position_weights = calculate_position_weights(train_data, num_classes=18)
    logger.info(f"Position weights: {position_weights}")
    logger.info(f"Min weight: {position_weights.min():.2f}, Max weight: {position_weights.max():.2f}, Mean: {position_weights.mean():.2f}")
    if args.use_lora:
        logger.info(
            f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}, targets={lora_targets}, "
            f"encoder_lr={args.encoder_lr}, head_lr={args.learning_rate}"
        )
    if args.attn_topk > 0:
        logger.info(f"Attention top-k pooling enabled: k={args.attn_topk}")
    logger.info(f"Window set: {args.window_set}")
    model = MultiAbnormalityClassifier(
        encoder_type=args.encoder_type,
        local_model_dir=args.encoder_dir,
        num_classes=18,
        position_weights=position_weights,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=lora_targets,
        encoder_lr=args.encoder_lr,
        attn_topk=args.attn_topk,
        window_set=args.window_set,
    )
    
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best-{epoch:02d}-{val_mean_auc:.4f}',
        monitor='val_mean_auc',
        mode='max',
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='lightning_logs',
        version=None,
        log_graph=True,
    )
    
    logger.info("Creating trainer...")
    
    # Optionally use early stopping after 6 epochs of no improvement on val_mean_auc
    #early_stopping_callback = EarlyStopping(
    #    monitor='val_mean_auc',
    #    mode='max',
    #    patience=6,
    #)
    # Add early_stopping_callback to the callbacks list 

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info(f"Training complete - best checkpoint saved to: {checkpoint_callback.best_model_path}")
    logger.info(f"Best val_mean_auc: {checkpoint_callback.best_model_score:.4f}")

    # load the best checkpoint and test the model
    best_model = MultiAbnormalityClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    trainer.test(best_model, test_loader)

if __name__ == '__main__':
    main()