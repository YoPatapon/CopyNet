import os
import random
import operator
import unicodedata

import torch
from torch.utils.data import Dataset

class Vocab(object):
    def __init__(self, lines, vocab_limit=-1):
        self.vocab = self.create_vocab(lines)
        if vocab_limit != -1:
            self.vocab = self.vocab[:vocab_limit]
            
        self.word2id = {
            '<pad>': 0, 
            '<s>': 1, 
            '</s>': 2, 
            '<unk>': 3}
        for idx, word in enumerate(self.vocab):
            self.word2id[word] = idx + 4
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        
    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)
          
    def create_vocab(self, lines):
        vocab = {}
        for line in lines:
            for word in line:
                # word = self.normalize(word)
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
                    
        if '<s>' in vocab:
            del vocab['<s>']
        if '<pad>' in vocab:
            del vocab['<pad>']
        if '</s>' in vocab:
            del vocab['</s>']
        if '<unk>' in vocab:
            del vocab['<unk>']
        
        sorted_word2id = sorted(
            vocab.items(), 
            key=operator.itemgetter(1), 
            reverse=True)
        sorted_words = [x[0] for x in sorted_word2id]
        return sorted_words
    
    def size(self):
        """ Return size of vocab. """
        return len(self.word2id)
    
    def tokens(self):
        """Get vocab tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.word2id.keys()
                  if k not in {'<pad>', '<unk>', '<s>', '</s>'}]
        return tokens
    
    
class SequencePairDataset(Dataset):
    def __init__(self, examples, vocab, args):
        self.args = args
        self.max_length = args.max_length
        self.examples = examples
        self.vocab = vocab
        self.use_extended_vocab = args.use_extended_vocab
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """ Vectorize a single example. """
        example = self.examples[idx]
        input_token_list = example['src'][:self.max_length]
        output_token_list = example['trg'][:self.max_length]
        
        # input_token_list = [self.vocab.normalize(token) for token in input_token_list]
        # output_token_list = [self.vocab.normalize(token) for token in output_token_list]
        
        input_seq = self.tokens_to_seq(input_token_list)
        output_seq = self.tokens_to_seq(output_token_list, input_tokens=input_token_list)
        
        if torch.cuda.is_available():
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()
        return input_seq, output_seq, ' '.join(input_token_list), ' '.join(output_token_list)
            
    def tokens_to_seq(self, tokens, input_tokens=None):
        """ Convert tokens to sequences of word id. """
        seq = torch.zeros(self.max_length).long()
        # seq = torch.LongTensor(self.max_length).zero_()
        seq_len = len(tokens)
        tok_to_idx_extension = dict()
        
        for pos, token in enumerate(tokens):
            if token in self.vocab.word2id:
                idx = self.vocab.word2id[token]
                
            elif token in tok_to_idx_extension:
                idx = tok_to_idx_extension[token]
                
            elif self.use_extended_vocab and input_tokens is not None:
                tok_to_idx_extension[token] = tok_to_idx_extension.get(
                    token, 
                    next((pos + len(self.vocab.word2id) 
                          for pos, input_token in enumerate(input_tokens) 
                          if input_token == token), 3))
                idx = tok_to_idx_extension[token]
                
            elif self.use_extended_vocab:
                idx = pos + len(self.vocab.word2id)
                
            else:
                idx = self.vocab.word2id['<unk>']
                
            seq[pos] = idx
            
        return seq
            
            