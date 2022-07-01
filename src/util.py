# util.py
# Gerenal utility module for various mathematical operations and training procedures.
# Modified from https://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import torch
import torch.nn as nn
import copy, re
from torch.autograd import Variable

def clones(module, N, T=1):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)] * T)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Pad and substitute generic <nat>/<var> tokens in minibatch.
def process_minibatch(x, pad, copy_vocab=None, tgt_vocab=None):
    batch = len(x)
    
    # Get the max seq size for the minibatch.
    max_length = 0
    for b in range(batch):
        seq_length = len(x[b])
        if seq_length > max_length:
            max_length = seq_length

    # Pad all examples with 0 to be equal to the max length.
    for b in range(batch):
        seq_length = len(x[b])
        if seq_length < max_length:
            x[b].extend(list(np.full(max_length - seq_length, pad, dtype=np.int32)))

    # Make the padded minibatch a homogeneous numpy array.
    x = np.concatenate(x).astype(np.int32).reshape(batch, -1)

    # If output, check for copied tokens and reference them with the copying vocab.
    # Substitute any copied <nat:+ and <var:+ tokens with generic <nat> and <var> tokens for embeddings.
    if copy_vocab is not None:
        vocab = len(tgt_vocab)
        for b in range(batch):
            nats = []
            vars = []
            for i in range(x.shape[1]):
                if x[b, i] >= vocab:
                    if re.match("<nat:.+", copy_vocab[b][x[b, i] - vocab]):
                        if not copy_vocab[b][x[b, i] - vocab] in nats:
                            nats.append(copy_vocab[b][x[b, i] - vocab])
                        x[b, i] = tgt_vocab.index("<nat" + str(min(nats.index(copy_vocab[b][x[b, i] - vocab]), 19)) + ">")
                    elif re.match("<var:.+", copy_vocab[b][x[b, i] - vocab]):
                        if not copy_vocab[b][x[b, i] - vocab] in vars:
                            vars.append(copy_vocab[b][x[b, i] - vocab])
                        x[b, i] = tgt_vocab.index("<var" + str(min(vars.index(copy_vocab[b][x[b, i] - vocab]), 19)) + ">")
                    elif re.match("<def:.+", copy_vocab[b][x[b, i] - vocab]):
                        x[b, i] = tgt_vocab.index("<def>")
                    else:
                        x[b, i] = tgt_vocab.index("<oov>")

    tensor_type = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    return torch.from_numpy(x).type(tensor_type)

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg, input_seqs, copy_vocab, tgt_vocab, pad=0):
        self.vocab = len(tgt_vocab)

        self.src = process_minibatch(src, pad)
        self.src_mask = (self.src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = process_minibatch(trg, pad)[:, :-1]
            self.trg_y = process_minibatch(trg, pad)[:, 1:]
            self.trg_a = process_minibatch(trg, pad, copy_vocab, tgt_vocab)[:, :-1]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
        self.input_seqs = process_minibatch(input_seqs, pad)
        self.copy_vocab = copy_vocab
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_batch(batch, model, loss_compute):
    # Run the model and compute the loss.
    memory = model.encode(batch.src, batch.src_mask)
    out = model.decode(memory, batch.src_mask, batch.trg_a, batch.trg_mask)
    loss = loss_compute(out, batch.trg_y, memory, batch.input_seqs, batch.ntokens)

    # Return the normalized loss.
    return loss / batch.ntokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        # Size is dynamic due to copying.
        self.size = x.size(1)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, memory, input_seq, norm):
        x = self.generator(x, memory, input_seq)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.data.item() * norm

def to_cuda(t):
    if torch.cuda.is_available():
        t = t.cuda()
    return t

def greedy_decode(model, src, src_mask, input_seq, copy_vocab, tgt_vocab, max_len):
    start_symbol = tgt_vocab.index("<start>")

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    model.generator.copy_weights = to_cuda(torch.zeros(1, 1, src.size(1)))
    for i in range(max_len-1):
        xs = process_minibatch(ys.tolist(), 0)
        xs_a = process_minibatch(ys.tolist(), 0, copy_vocab, tgt_vocab)
        out = model.decode(memory, src_mask, 
                           Variable(xs_a), 
                           Variable(subsequent_mask(xs.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1], memory[-1, :], input_seq)
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0, 0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
