"""A tiny GPT that honours packed-batch masks (document-masked attention + position ids)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, L, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, seq_len: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, tokens, position_ids, attn_mask):
        x = self.tok(tokens) + self.pos(position_ids)
        for b in self.blocks:
            x = b(x, attn_mask)
        return self.head(self.ln(x))


def token_losses(model, arrays: dict) -> torch.Tensor:
    """Per-token cross entropy [B, L], zero where loss_mask == 0."""
    from .packing import attention_mask
    t = torch.from_numpy(arrays["tokens"])
    pos = torch.from_numpy(arrays["position_ids"])
    mask = torch.from_numpy(attention_mask(arrays["segment_ids"]))
    logits = model(t, pos, mask)
    lab = torch.from_numpy(arrays["labels"])
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1),
                         reduction="none").view(lab.shape)
    return ce * torch.from_numpy(arrays["loss_mask"]).to(ce.dtype)
