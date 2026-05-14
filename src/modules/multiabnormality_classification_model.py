# multiabnormality_classification_model.py
# purpose: multi-abnormality classification model using a pretrained encoder and custom classification head
# author: Anthony Smaldore

import logging
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, hamming_loss, average_precision_score

from modules.encoder import create_encoder
from modules.classification_head import GatedAttentionMILHead

logger = logging.getLogger(__name__)

# class map for the 18 abnormality classes
CLASS_NAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

class MultiAbnormalityClassifier(pl.LightningModule):
    """
    Multi-abnormality classifier for CT-RATE challenge.
    
    Architecture:
    Step 1. Frozen DinoV3 encoder extracts features from each 2D slice
    Step 2. Gated attention MIL head computes per-class attention weights over slices
    Step 3. Each class pools slice features with its own attention distribution, then classifies with a per-class linear layer
    
    Args:
        encoder_type: Type of encoder ('dinov3')
        local_model_dir: Path to local model directory
        num_classes: Number of abnormality classes (default: 18)
        dropout: Dropout probability in classification head (default: 0.3)
        learning_rate: Learning rate for optimizer (default: 1e-3)
    """
    
    def __init__(
        self, 
        encoder_type='dinov3',
        local_model_dir=None,
        num_classes=18,
        position_weights=None,
        dropout=0.3,
        learning_rate=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        if local_model_dir is None:
            raise ValueError("local_model_dir must be provided to load the encoder")

        if position_weights is not None:
            self.register_buffer('position_weights', position_weights)
        else:
            self.register_buffer('position_weights', torch.ones(num_classes))
        self.register_buffer('thresholds', torch.full((num_classes,), 0.5))

        
        # load frozen encoder and trainable classification head
        self.encoder = create_encoder(encoder_type, local_model_dir=local_model_dir)
        self.head = GatedAttentionMILHead(
            embedding_dim=self.encoder.embedding_dim,
            num_classes=num_classes,
            attn_dim=128,
            dropout=dropout,
        )
        
        # loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.position_weights)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        """
        Forward pass through multi-abnormality classification head
        
        Args:
            x: Input tensor of shape [B, 3, D, H, W]
               where B=batch, 3=channels, D=depth, H=224, W=224
        
        Returns:
            logits: Output tensor of shape [B, num_classes]
        """
        B, C, D, H, W = x.shape
        
        # flatten batch and depth dimensions and extract embeddings from all slices via encoder
        x_slices = x.permute(0, 2, 1, 3, 4)
        x_slices = x_slices.contiguous().view(B * D, C, H, W)
    
        slice_embeddings = self.encoder(x_slices)
        
        # reshape back to separate batch and depth and pass through classification head
        slice_embeddings = slice_embeddings.view(B, D, -1)
        logits = self.head(slice_embeddings)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """
        Training step for one batch - required by PyTorch Lightning
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of current batch
        
        Returns:
            loss: Training loss for this batch
        """
        images = batch["image"]
        labels = batch["label"]
        
        # forward pass and loss computation
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch - required by PyTorch Lightning for validation 
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of current batch
        """
        images = batch["image"]
        labels = batch["label"]
        
        # forward pass and loss computation
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # convert logits to predictions
        probs = torch.sigmoid(logits)
        preds = (probs > self.thresholds).float()
        
        # store outputs for validation metrics and log validation loss
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds.detach().cpu(),
            'probs': probs.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """
        Test step for one batch - required by PyTorch Lightning for testing
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of current batch
        """
        images = batch["image"]
        labels = batch["label"]
        
        # forward pass and loss computation
        logits = self(images)
        loss = self.criterion(logits, labels)

        # convert logits to predictions
        probs = torch.sigmoid(logits)
        preds = (probs > self.thresholds).float()
        
        # store outputs for test metrics and log test loss
        self.test_step_outputs.append({
            'loss': loss,
            'preds': preds.detach().cpu(),
            'probs': probs.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        """
        Compute epoch-level validation metrics - PyTorch Lightning
        
        Aggregates predictions across all validation batches and computes:
        - Mean AUC across all classes
        - Accuracy, F1, Precision, Recall
        """
        # concatenate all batch outputs and convert to numpy for sklearn metrics
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        preds_np = all_preds.numpy()
        probs_np = all_probs.numpy()
        labels_np = all_labels.numpy()
        
        try:
            # per-class F1-optimal threshold tuning
            grid = np.linspace(0.05, 0.95, 19)
            best_t = np.full(self.hparams.num_classes, 0.5)
            for i in range(self.hparams.num_classes):
                if len(np.unique(labels_np[:, i])) > 1:
                    f1s = [
                        f1_score(labels_np[:, i],
                                (probs_np[:, i] > t).astype(int),
                                zero_division=0)
                        for t in grid
                    ]
                    best_t[i] = grid[int(np.argmax(f1s))]
            self.thresholds = torch.tensor(best_t, dtype=torch.float32, device=self.device)
            preds_np = (probs_np > best_t).astype(np.float32)

            # compute the per class AUC scores for the 18 abnormality classes and 
            # the mean AUC score across all classes
            per_class_auc = {}

            for i in range(self.hparams.num_classes):
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
                if len(np.unique(labels_np[:, i])) > 1:
                    per_class_auc[name] = float(roc_auc_score(labels_np[:, i], probs_np[:, i]))
                else:
                    per_class_auc[name] = float('nan')

            valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
            mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

            for name, v in per_class_auc.items():
                if not np.isnan(v):
                    self.log(f'val_auc/{name}', v, on_epoch=True)

            # compute the per class Average Precision scores for the 18 abnormality classes and 
            # the mean Average Precision score across all classes
            per_class_ap = {}

            for i in range(self.hparams.num_classes):
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
                if len(np.unique(labels_np[:, i])) > 1:
                    per_class_ap[name] = float(average_precision_score(labels_np[:, i], probs_np[:, i]))
                else:
                    per_class_ap[name] = float('nan')

            valid_aps = [v for v in per_class_ap.values() if not np.isnan(v)]
            mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

            for name, v in per_class_ap.items():
                if not np.isnan(v):
                    self.log(f'val_ap/{name}', v, on_epoch=True)
            
            # accuracy, F1, precision, recall, and hamming loss
            accuracy = accuracy_score(labels_np.flatten(), preds_np.flatten())
            f1 = f1_score(labels_np, preds_np, average='micro', zero_division=0)
            precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
            recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
            hamming = hamming_loss(labels_np, preds_np)

            # log validation metrics
            self.log('val_mean_auc', mean_auc, prog_bar=True)
            self.log('val_mean_ap', mean_ap, prog_bar=True)
            self.log('val_accuracy', accuracy, prog_bar=True)
            self.log('val_f1', f1, prog_bar=True)
            self.log('val_precision', precision)
            self.log('val_recall', recall)
            self.log('val_hamming_loss', hamming)
            
        except Exception as e:
            logger.error(f"Error computing validation metrics: {e}")
        
        # clear outputs for next epoch
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """
        Compute epoch-level test metrics - PyTorch Lightning
            
        Aggregates predictions across all test batches and computes:
        - Mean AUC across all classes
        - Mean Average Precision
        - Accuracy, F1, Precision, Recall, and Hamming Loss
        """
        # concatenate all batch outputs and convert to numpy for sklearn metrics
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        preds_np = all_preds.numpy()
        probs_np = all_probs.numpy()
        labels_np = all_labels.numpy()
        
        try:
            # compute the per class AUC scores for the 18 abnormality classes and 
            # the mean AUC score across all classes
            per_class_auc = {}

            for i in range(self.hparams.num_classes):
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
                if len(np.unique(labels_np[:, i])) > 1:
                    per_class_auc[name] = float(roc_auc_score(labels_np[:, i], probs_np[:, i]))
                else:
                    per_class_auc[name] = float('nan')

            valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
            mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

            for name, v in per_class_auc.items():
                if not np.isnan(v):
                    self.log(f'test_auc/{name}', v, on_epoch=True)

            # compute the per class Average Precision scores for the 18 abnormality classes and 
            # the mean Average Precision score across all classes
            per_class_ap = {}

            for i in range(self.hparams.num_classes):
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
                if len(np.unique(labels_np[:, i])) > 1:
                    per_class_ap[name] = float(average_precision_score(labels_np[:, i], probs_np[:, i]))
                else:
                    per_class_ap[name] = float('nan')

            valid_aps = [v for v in per_class_ap.values() if not np.isnan(v)]
            mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

            for name, v in per_class_ap.items():
                if not np.isnan(v):
                    self.log(f'test_ap/{name}', v, on_epoch=True)
            
            # accuracy, F1, precision, recall, and hamming loss
            accuracy = accuracy_score(labels_np.flatten(), preds_np.flatten())
            f1 = f1_score(labels_np, preds_np, average='micro', zero_division=0)
            precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
            recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
            hamming = hamming_loss(labels_np, preds_np)

            # log the per class AUC and AP scores for the 18 abnormality classes and the mean AUC and AP scores across all classes
            logger.info(f"{'Class':<42}  {'AUC':>6}  {'AP':>6}  {'n_pos':>5}")
            rows = []
            for i in range(self.hparams.num_classes):
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}'
                n_pos = int(labels_np[:, i].sum())
                rows.append((name, per_class_auc[name], per_class_ap[name], n_pos))
                
            # sort by AUC descending
            rows.sort(key=lambda r: (-(r[1]) if not np.isnan(r[1]) else float('inf')))
            for name, auc, ap, n_pos in rows:
                auc_s = f"{auc:.3f}" if not np.isnan(auc) else "  N/A"
                ap_s  = f"{ap:.3f}"  if not np.isnan(ap)  else "  N/A"
                logger.info(f"{name:<42}  {auc_s:>6}  {ap_s:>6}  {n_pos:>5}")
            logger.info(f"{'MEAN (valid only)':<42}  {mean_auc:>6.3f}  {mean_ap:>6.3f}  "
                        f"{int(labels_np.sum()):>5}")

            # log rest of the metrics 
            self.log('test_mean_auc', mean_auc, prog_bar=True)
            self.log('test_mean_ap', mean_ap, prog_bar=True)
            self.log('test_accuracy', accuracy, prog_bar=True)
            self.log('test_f1', f1, prog_bar=True)
            self.log('test_precision', precision)
            self.log('test_recall', recall)
            self.log('test_hamming_loss', hamming)
            
        except Exception as e:
            logger.error(f"Error computing test metrics: {e}")
        
        # clear outputs for next test run
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        
        Returns:
            Optimizer and learning rate scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-2,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
