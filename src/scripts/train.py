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

# add src to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.data_loader import create_subset_manifest
from modules.data_loader import create_train_val_test_splits
from modules.transforms import get_train_transforms, get_valid_transforms
from modules.multiabnormality_classification_model import MultiAbnormalityClassifier

logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments to set the training parameters and output specifications
    Returns:
        args: Parsed command line arguments
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
    
    # additional settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training (rest split equally for val/test)')
    
    return parser.parse_args()



def prepare_data(args):
    """
    Create manifest and split into train/val/test sets
    Args:
        args: Parsed command line arguments
    Returns:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
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
    Create MONAI datasets and PyTorch dataloaders
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        args: Parsed command line arguments
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
    """
    
    # create the training and validation transforms with the get_train_transforms and get_valid_transforms functions
    train_transforms = get_train_transforms(
        spatial_size=(96, 224, 224),
        pixdim=(1.5, 1.5, 1.5)
    )
    
    val_transforms = get_valid_transforms(
        spatial_size=(96, 224, 224),
        pixdim=(1.5, 1.5, 1.5)
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
    Calculate position weights for BCEWithLogitsLoss function to handle class imbalance

    Args:
        train_dataset: Training dataset
        num_classes: Number of classes
    Returns:
        position_weights: Position weights
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
    Main function to train the multi-abnormality classification model
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
    model = MultiAbnormalityClassifier(
        encoder_type=args.encoder_type,
        local_model_dir=args.encoder_dir,
        num_classes=18,
        position_weights=position_weights,
        dropout=args.dropout,
        learning_rate=args.learning_rate
    )
    
    # setup the checkpoint callback to save the best model based on the validation accuracy
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

    # setup the TensorBoard logger for tracking model perfromance and other visualizations 
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='lightning_logs',
        version=None,
        log_graph=True,
    )
    
    # create the early stopping and checkpoint callbacks with trainer from the PyTorch Lightning Trainer class to train the model
    logger.info("Creating trainer...")
    
    # optionally use early stopping after 6 epochs of no improvement in the validation mean AUC score
    #early_stopping_callback = EarlyStopping(
    #    monitor='val_mean_auc',
    #    mode='max',
    #    patience=6,
    #)
    # add early_stopping_callback to the callbacks list 

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
    
    # load the data into the model, train it, and display the best model path and validation accuracy
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