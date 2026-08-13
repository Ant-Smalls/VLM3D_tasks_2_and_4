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

    Optional top-k sparsification: after softmax over slices, keep only the
    top-k weights per class, zero the rest, and renormalize (attn_topk > 0).
    """

    def __init__(self, embedding_dim=768, num_classes=18,
                 attn_dim=128, dropout=0.3, attn_topk=0):
        super().__init__()
        self.V = nn.Linear(embedding_dim, attn_dim)
        self.U = nn.Linear(embedding_dim, attn_dim)
        self.w = nn.Linear(attn_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        # 0 disables top-k (full softmax pooling, matching the frozen-600 baseline)
        self.attn_topk = int(attn_topk) if attn_topk is not None else 0

    def _maybe_topk(self, a):
        """
        Keep top-k attention weights along the slice dimension, renormalize.

        Args:
            a: Softmax attention [B, S, C]
        Returns:
            a: Possibly sparsified attention [B, S, C]
        """
        if self.attn_topk <= 0:
            return a
        s = a.shape[1]
        k = min(self.attn_topk, s)
        # keep scores on the top-k slices; zero others; renormalize over slices
        _, topk_idx = torch.topk(a, k=k, dim=1)
        mask = torch.zeros_like(a)
        mask.scatter_(1, topk_idx, 1.0)
        a = a * mask
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return a

    def forward(self, x, return_features=False, return_attention=False):
        """
        Args:
            x: Slice embeddings [B, S, D]
            return_features: If True, also return class-specific pooled embeddings z [B, C, D]
            return_attention: If True, also return per-slice attention weights a [B, S, C]
        """
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        a = self.w(v * u)
        a = torch.softmax(a, dim=1)
        a = self._maybe_topk(a)
        z = torch.einsum('bsc,bsd->bcd', a, x)
        z = self.dropout(z)
        logits = (self.classifier.weight * z).sum(-1) + self.classifier.bias

        # optional diagnostic returns (attention is post-softmax, post top-k if enabled)
        if return_features and return_attention:
            return logits, z, a
        if return_features:
            return logits, z
        if return_attention:
            return logits, a
        return logits
