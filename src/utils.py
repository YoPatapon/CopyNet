"""CopyNet reader utilities."""

import json
import time
import logging
import string
import regex as re

from collections import Counter
from data import Vocab

import torch
from torch.autograd import Variable

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

bos = '<s>'
eos = '</s>'

def load_data(filename, add_sql_symbol=False, add_query_symbol=True):
    """ Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]
        
    # Add symbol tokens to sql or query
    if add_sql_symbol or add_query_symbol:
        for ex in examples:
            if add_sql_symbol:
                ex['src'] = [bos] + ex['src']
                ex['src'] = ex['src'] + [eos]
            if add_query_symbol:
                ex['trg'] = [bos] + ex['trg']
                ex['trg'] = ex['trg'] + [eos]
                
    return examples
  
    
def build_vocab(exs, args):
    """ Build vocab using training examples. """
    src_lines, tgt_lines = [], []
    
    for ex in exs:
        src_lines.append(ex['src'])
        tgt_lines.append(ex['trg'])
        
    vocab = Vocab(
        src_lines + tgt_lines, 
        args.vocab_limit)
    return vocab


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
    
    
def to_np(x):
    return x.data.cpu().numpy()

def trim_seqs(seqs):
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for idx in to_np(output_seq):
            trimmed_seq.append(idx[0])
            if idx == 2:
                break
        trimmed_seqs.append(trimmed_seq)
    return trimmed_seqs

def trim_seq_probs(seqs, probs):
    trimmed_seqs, trimmed_probs = [], []
    for output_seq, output_prob in zip(seqs, probs):
        trimmed_seq, trimmed_prob = [], []
        for idx, p in zip(to_np(output_seq), to_np(output_prob)):
            trimmed_seq.append(idx[0])
            trimmed_prob.append(p[0])
            if idx == 2:
                break
        trimmed_seqs.append(trimmed_seq)
        trimmed_probs.append(trimmed_prob)
    return trimmed_seqs, trimmed_probs


def seq_to_string(seq, idx_to_tok, input_tokens=None):
    vocab_size = len(idx_to_tok)
    seq_length = (seq != 0).sum()
    words = []
    for idx in seq[:seq_length]:
        if idx < vocab_size:
            words.append(idx_to_tok[idx])
        elif input_tokens is not None:
            words.append(input_tokens[idx - vocab_size])
        else:
            words.append('<???>')
    string = ' '.join(words)
    return string

def token_list_transpose(tokens_list):
    """ Token list from data_loader is transposed, we need to transform them back. """
    
    seq_length = len(tokens_list)
    batch_size = len(tokens_list[0])
    
    correct_tokens = [['' for i in range(seq_length)] for j in range(batch_size)]
    for i, tokens in enumerate(tokens_list):
        for j, token in enumerate(tokens):
            correct_tokens[j][i] = token
    
    return correct_tokens
    
    
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


