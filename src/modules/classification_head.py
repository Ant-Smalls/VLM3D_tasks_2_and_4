# classification_head.py
# purpose: gated attention MIL head for multi-label abnormality classification
# author: Anthony Smaldore

import torch
import torch.nn as nn


class GatedAttentionMILHead(nn.Module):
    """
    Gated attention-based MIL pooling over slice embeddings
    Produces a class-specific weighted sum of slice features, then a per-class
    linear classifier.
    """

    def __init__(self, embedding_dim=768, num_classes=18,
                 attn_dim=128, dropout=0.3):
        super().__init__()
        self.V = nn.Linear(embedding_dim, attn_dim)
        self.U = nn.Linear(embedding_dim, attn_dim)
        self.w = nn.Linear(attn_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        a = self.w(v * u)                           
        a = torch.softmax(a, dim=1)
        z = torch.einsum('bsc,bsd->bcd', a, x)      
        z = self.dropout(z)
        logits = (self.classifier.weight * z).sum(-1) + self.classifier.bias
        return logits                               