""" Implementation of the copynet. """

import sys
import time
import math
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from component import EncoderRNN, CopyNetDecoder, AttentionDecoder

logger = logging.getLogger(__name__)

class CopyNet(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    
    def __init__(self, args, vocab):
        super(CopyNet, self).__init__()
        
        # Store config
        self.args = args
        self.vocab = vocab
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.hidden_size
        self.encoder = EncoderRNN(self.vocab.size(),
                                  self.hidden_size,
                                  self.embedding_dim)
        self.decoder_type = args.decoder_type
        self.max_length = args.max_length
        
        decoder_hidden_size = 2 * self.encoder.hidden_size
        if self.decoder_type == 'attn':
            self.decoder = AttentionDecoder(decoder_hidden_size,
                                            self.embedding_dim,
                                            self.vocab,
                                            self.max_length)
        elif self.decoder_type == 'copy':
            self.decoder = CopyNetDecoder(decoder_hidden_size,
                                          self.embedding_dim,
                                          self.vocab,
                                          self.max_length)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")
            
        self.citerion = nn.NLLLoss(ignore_index=0)
        self.updates = 0
        
        
    def forward(self, inputs, lengths, targets=None):
        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     targets=targets)
        return decoder_outputs, sampled_idxs
    
    def get_query_prob(self, inputs, lengths, targets):
        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        
        decoder_outputs, sampled_idxs, query_probs = self.decoder.fetch_decode_probs(encoder_outputs,
                                                                                     inputs,
                                                                                     hidden,
                                                                                     targets)
        return decoder_outputs, sampled_idxs, query_probs
    
    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.encoder.embedding.parameters():
                p.requires_grad = False
            for p in self.decoder.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
            
    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.vocab.word2id}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        e_embedding = self.encoder.embedding.weight.data
        d_embedding = self.decoder.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == e_embedding.size(1) + 1 and len(parsed) == d_embedding.size(1) + 1)
                # w = self.vocab.normalize(parsed[0])
                w = parsed[0]
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        e_embedding[self.vocab.word2id[w]].copy_(vec)
                        d_embedding[self.vocab.word2id[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        e_embedding[self.vocab.word2id[w]].add_(vec)
                        d_embedding[self.vocab.word2id[w]].add_(vec)

        for w, c in vec_counts.items():
            e_embedding[self.vocab.word2id[w]].div_(c)
            d_embedding[self.vocab.word2id[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))