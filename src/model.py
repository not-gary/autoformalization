# model.py
# Transformer encode/decoder model implementations.
# Modified from https://nlp.seas.harvard.edu/2018/04/03/attention.html

from attention import *
from util import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt_a, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt_a, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt_a, tgt_mask):
        return self.decoder(self.tgt_embed(tgt_a), memory, src_mask, tgt_mask)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, T):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N, T)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, T):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N, T)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, src_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(src_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.state_size = d_model
        self.vocab_size = vocab
        self.Wo = nn.Linear(self.state_size, self.vocab_size)
        self.Wc = nn.Linear(self.state_size, self.state_size)

    def forward(self, s, h, input_seq):
        # We might be missing the batch dimension. Add that in.
        if s.dim() < 3:
            s = s.unsqueeze(0)
            h = h.unsqueeze(0)

        batch_size = s.size(0)
        
        # Find psi_g.
        gen_scores = self.Wo(s) # [batch x input x vocab]

        # Find psi_c.
        copy_scores = torch.tanh(self.Wc(h)) # [batch x input x state]
        copy_scores = torch.bmm(copy_scores, torch.transpose(s, -1, -2)) # [batch x input x input]

        # Move our scores to the last dim.
        copy_scores = torch.transpose(copy_scores, -1, -2) # [batch x input x input]

        # Combine our scores together and softmax.
        scores = torch.cat([gen_scores, copy_scores], dim=-1) # [batch x input x vocab + input]
        scores = F.softmax(scores, dim=-1)

        # Briefly split the probabilities so we can perform the following summation.
        gen_probs = scores[:, :, :self.vocab_size] # [batch x input x vocab]
        copy_probs = scores[:, :, self.vocab_size:] # [batch x input x input]

        # Sum over matching tokens in the input.
        one_hot = F.one_hot(input_seq).float()
        copy_probs = torch.bmm(copy_probs, one_hot) # [batch x input x input]

        probs = torch.cat([gen_probs, copy_probs], dim=-1)

        log_probs = torch.log(probs + 1e-20)
        return log_probs


def make_model(src_vocab, tgt_vocab, N=6, T=1,
               d_model=512, d_ff=2048, h=8, k=5, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        Encoder(EncoderLayer(
                d_model,
                MultiHeadedRelativeAttention(h, d_model, k, dropout),
                PositionwiseFeedForward(d_model, d_ff, dropout),
                dropout),
            N, T),
        Decoder(DecoderLayer(
                d_model,
                MultiHeadedRelativeAttention(h, d_model, k, dropout),
                MultiHeadedAttention(h, d_model), 
                PositionwiseFeedForward(d_model, d_ff, dropout),
                dropout),
            N, T),
        Embeddings(d_model, src_vocab),
        Embeddings(d_model, tgt_vocab),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
