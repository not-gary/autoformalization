# attention.py
# Multi-headed attention implementation module.
# Modified from https://nlp.seas.harvard.edu/2018/04/03/attention.html

from util import clones, to_cuda

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def _attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self._attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedRelativeAttention(nn.Module):
    def __init__(self, h, d_model, k, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedRelativeAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.k = k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.wK = nn.Parameter(torch.empty(2 * k + 1, self.d_k))
        self.wV = nn.Parameter(torch.empty(2 * k + 1, self.d_k))
        
        # Initialize our relation weights.
        nn.init.xavier_uniform_(self.wK)
        nn.init.xavier_uniform_(self.wV)

    def _relative_dot_product(self, x, a):
        b = x.size(0)
        h = x.size(1)
        l = x.size(2)

        rpr_scores = torch.matmul(x.permute(2, 0, 1, 3).reshape(l, b * h, -1), a)
        rpr_scores = rpr_scores.reshape(l, b, h, -1).permute(1, 2, 0, 3)

        return rpr_scores

    def _toeplitz(self, length, weights):
        a = torch.roll(weights[:self.k + 1], 1, 0)
        a = torch.cat([a, a[-1].repeat(length - (self.k + 1), 1)], dim=0) if length > self.k + 1 else a[:length]

        b = weights[self.k:]
        b = torch.cat([b, b[-1].repeat(length - (self.k + 1), 1)], dim=0) if length > self.k + 1 else b[:length]

        vals = torch.cat([b, a[1:].flip(0)])
        shape = len(a), len(b)
        i, j = to_cuda(torch.ones(*shape)).nonzero().T
        return vals[j - i].reshape(*shape, weights.size(-1))

    def _attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        length = query.size(-2)

        # Create clamped matrices using trained attention weights.
        aK = self._toeplitz(length, self.wK)
        aV = self._toeplitz(length, self.wV)

        rpr_scores_K = self._relative_dot_product(query, torch.transpose(aK, -2, -1))
        
        scores = (torch.matmul(query, key.transpose(-2, -1)) + rpr_scores_K) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        rpr_scores_V = self._relative_dot_product(p_attn, aV)
        return torch.matmul(p_attn, value) + rpr_scores_V, p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self._attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
