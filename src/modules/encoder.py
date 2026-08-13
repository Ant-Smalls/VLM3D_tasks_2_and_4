# encoder.py
# purpose: frozen encoders for extracting image embeddings
# author: Anthony Smaldore

import logging
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# default LoRA target modules for Hugging Face Dinov2 attention (Q/K/V)
DEFAULT_LORA_TARGETS = ["query", "key", "value"]


class DinoV2Encoder(nn.Module):
    """
    DinoV2 encoder for extracting image embeddings
    
    Loads pretrained DinoV2 model. By default all backbone weights stay frozen.
    Optional LoRA adapters (peft) on attention Q/K/V enable light finetuning.
    
    Args:
        local_model_dir: Path to local model directory
        use_lora: If True, attach LoRA adapters and allow their gradients
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        lora_targets: Module name list for peft target_modules
    """
    
    def __init__(
        self,
        local_model_dir,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_targets=None,
    ):
        super(DinoV2Encoder, self).__init__()
        
        logger.info(f"Loading DinoV2 encoder from local directory: {local_model_dir}...")
        
        # load the Hugging Face formatted model from local directory
        self.backbone = AutoModel.from_pretrained(local_model_dir)
        self.use_lora = use_lora
        
        # CLS + mean-patch concatenation → 2 * hidden_size (768 for dinov2-small)
        self.embedding_dim = self.backbone.config.hidden_size * 2
        
        # freeze all backbone parameters (LoRA will re-enable adapter grads below)
        for param in self.backbone.parameters():
            param.requires_grad = False

        if use_lora:

            if lora_targets is None:
                lora_targets = list(DEFAULT_LORA_TARGETS)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                f"DinoV2 encoder loaded with LoRA "
                f"(r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, "
                f"targets={list(lora_targets)}; "
                f"{trainable:,} trainable / {total:,} total backbone params; "
                f"{self.embedding_dim}-dim embeddings)"
            )
        else:
            logger.info(f"DinoV2 encoder loaded and frozen ({self.embedding_dim}-dim embeddings)")
    
    def forward(self, x):
        """
        Extract embeddings from input images
        
        Args:
            x: Input tensor of shape [B, 3, 224, 224]
        
        Returns:
            embeddings: Output tensor of shape [B, embedding_dim]
        """
        # frozen path: no autograd through backbone; LoRA path needs grads on adapters
        if self.use_lora:
            outputs = self.backbone(pixel_values=x)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0, :]
            patch_mean = hidden[:, 1:, :].mean(dim=1)
            embeddings = torch.cat([cls, patch_mean], dim=-1)
        else:
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



def create_encoder(
    encoder_type='dinov2',
    local_model_dir=None,
    use_lora=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_targets=None,
):
    """
    Create encoder instance based on the encoder type
    
    Args:
        encoder_type: Type of encoder to create ('dinov2' or 'dinov3')
                    choices: ['dinov2', 'dinov3']
        local_model_dir: Path to local model directory
        use_lora: Attach LoRA adapters (dinov2 only)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        lora_targets: Module name list for peft target_modules
    
    Returns:
        Encoder instance
    """
    if use_lora and encoder_type != 'dinov2':
        raise ValueError(
            f"LoRA is only supported for dinov2 (got encoder_type={encoder_type!r})"
        )

    if encoder_type == 'dinov2':
        if local_model_dir is None:
            raise ValueError("Must provide local_model_dir for DinoV2 encoder.")
        return DinoV2Encoder(
            local_model_dir=local_model_dir,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_targets=lora_targets,
        )

    elif encoder_type == 'dinov3':
        if local_model_dir is None:
            raise ValueError("Must provide local_model_dir for DinoV3 encoder.")
        return DinoV3Encoder(local_model_dir=local_model_dir)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Available: ['dinov2']")
