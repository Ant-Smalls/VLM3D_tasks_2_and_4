# encoder.py
# purpose: frozen encoders for extracting image embeddings
# author: Anthony Smaldore

import logging
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class DinoV2Encoder(nn.Module):
    """
    Frozen DinoV2 encoder for extracting image embeddings
    
    Loads pretrained DinoV2 model and freezes all parameters for feature extraction
    
    Args:
        local_model_dir: Path to local model directory
    """
    
    def __init__(self, local_model_dir):
        super(DinoV2Encoder, self).__init__()
        
        logger.info(f"Loading DinoV2 encoder from local directory: {local_model_dir}...")
        
        # load the Hugging Face formatted model from local directory
        self.backbone = AutoModel.from_pretrained(local_model_dir)
        
        # CLS + mean-patch concatenation → 2 * hidden_size (768 for dinov2-small)
        self.embedding_dim = self.backbone.config.hidden_size * 2
        
        # freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        logger.info(f"DinoV2 encoder loaded and frozen ({self.embedding_dim}-dim embeddings)")
    
    def forward(self, x):
        """
        Extract embeddings from input images
        
        Args:
            x: Input tensor of shape [B, 3, 224, 224]
        
        Returns:
            embeddings: Output tensor of shape [B, embedding_dim]
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0, :]
            patch_mean = hidden[:, 1:, :].mean(dim=1)
            embeddings = torch.cat([cls, patch_mean], dim=-1)

        return embeddings

class DinoV3Encoder(nn.Module):
    """
    Frozen DinoV3 encoder for extracting image embeddings
    
    Loads pretrained DinoV3 model and freezes all parameters for feature extraction
    
    Args:
        local_model_dir: Path to local model directory
    """
    
    def __init__(self, local_model_dir):
        super(DinoV3Encoder, self).__init__()
        
        logger.info(f"Loading DinoV3 encoder from local directory: {local_model_dir}...")
        
        # load the Hugging Face formatted model from local directory
        self.backbone = AutoModel.from_pretrained(local_model_dir)
        
        # dynamically get the embedding dimension (384 for dinov3-small)
        self.embedding_dim = self.backbone.config.hidden_size * 2
        
        # freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        logger.info(f"DinoV3 encoder loaded and frozen ({self.embedding_dim}-dim embeddings)")
    
    def forward(self, x):
        """
        Extract embeddings from input images
        
        Args:
            x: Input tensor of shape [B, 3, 224, 224]
        
        Returns:
            embeddings: Output tensor of shape [B, embedding_dim]
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0, :]
            patch_mean = hidden[:, 1:, :].mean(dim=1)
            embeddings = torch.cat([cls, patch_mean], dim=-1)
            
        return embeddings



def create_encoder(encoder_type='dinov2', local_model_dir=None):
    """
    Create encoder instance based on the encoder type
    
    Args:
        encoder_type: Type of encoder to create ('dinov2' or 'dinov3')
                    choices: ['dinov2', 'dinov3']
        local_model_dir: Path to local model directory
    
    Returns:
        Encoder instance
    """
    if encoder_type == 'dinov2':
        if local_model_dir is None:
            raise ValueError("Must provide local_model_dir for DinoV2 encoder.")
        return DinoV2Encoder(local_model_dir=local_model_dir)

    elif encoder_type == 'dinov3':
        if local_model_dir is None:
            raise ValueError("Must provide local_model_dir for DinoV3 encoder.")
        return DinoV3Encoder(local_model_dir=local_model_dir)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Available: ['dinov2']")
