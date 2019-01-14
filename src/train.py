""" Main CopyNet training script. """

import argparse
import os
import sys
import json
import torch
import logging
import subprocess

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary

from utils import *
from model import CopyNet
from data import SequencePairDataset

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logger = logging.getLogger()

# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------

# Defaults
ROOT_DIR = "./data"
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')
MODEL_DIR = './models/checkpoints/logs'
EMBED_DIR = os.path.join(ROOT_DIR, 'embeddings')

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--dev-batch-size', type=int, default=16,
                         help='Batch size for training')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='train_processed.jsonl',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='dev_processed.jsonl',
                       help='Preprocessed dev file')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--display-samples', type='bool', default=True,
                         help='Display top 5 samples in validation process.')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')
    
    # Model architecture
    model = parser.add_argument_group('Model Architecture')
    model.add_argument('--decoder-type', type=str, default='copy',
                       help='Decoder architecture type')
    model.add_argument('--vocab-limit', type=int, default=-1,
                       help='Limit for vocab creation')
    model.add_argument('--use-extended-vocab', type='bool', default=True,
                       help='Use extended vocab for copying')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden size of RNN units')
    model.add_argument('--rnn-type', type=str, default='lstm',
                       help='RNN type: LSTM, GRU, or RNN')

    # Optimization details
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: sgd or adam')
    optim.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    optim.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix-embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--rnn-padding', type='bool', default=False,
                       help='Explicitly account for padding in RNN encoding')
    optim.add_argument('--max-length', type=int, default=30,
                       help='The max span allowed during decoding')
    
    
def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.pt')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new vocab."""
    
    # Build a dictionary from the data sqls+queries (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build vocab')
    
    vocab = build_vocab(train_exs + dev_exs, args)
    logger.info('Num words = %d' % vocab.size())
    
    # Initialize model
    model = CopyNet(args, vocab)
    logger.info('-' * 100)
    logger.info('Model Architecture')
    logger.info(model)
    if args.embedding_file:
        model.load_embeddings(vocab.tokens(), args.embedding_file)
    
    return model, vocab


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = AverageMeter()
    epoch_time = Timer()
    
    for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(data_loader):
        # input_idxs and target_idxs have dim (batch_size x max_len)
        # they are NOT sorted by length

        lengths = (input_idxs != 0).long().sum(dim=1)
        sorted_lengths, order = torch.sort(lengths, descending=True)

        input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
        target_variable = Variable(target_idxs[order, :])
        
        model.optimizer.zero_grad()
        output_log_probs, output_ses = model(input_variable,
                                             list(sorted_lengths),
                                             targets=target_variable)
        
        batch_size = input_variable.shape[0]
        flattened_outputs = output_log_probs.view(batch_size * model.max_length, -1)
        
        batch_loss = model.citerion(flattened_outputs, target_variable.contiguous().view(-1))
        batch_loss.backward()
        model.optimizer.step()
        
        model.updates += 1
        
        train_loss.update(batch_loss[0], batch_size)
        
        if batch_idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], batch_idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
            
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
            (global_stats['epoch'], epoch_time.time()))
        
        # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)
            
            
def validate(args, data_loader, model, vocab, global_stats):
    """ Run one full validation process. """
    eval_time = Timer()
    val_loss = AverageMeter()
    
    all_output_seqs = []
    all_target_seqs = []
    all_input_seqs = []
    
    for batch_idx, (input_idxs, target_idxs, input_tokens, _) in enumerate(data_loader):
        
        input_lengths = (input_idxs != 0).long().sum(dim=1)
        sorted_lengths, order = torch.sort(input_lengths, descending=True)
        
        # No grad mode
        with torch.no_grad():
        
            input_variable = Variable(input_idxs[order, :][:, :max(input_lengths)], requires_grad=False)
            target_variable = Variable(target_idxs[order, :], requires_grad=False)
            batch_size = input_variable.shape[0]
            # Sort the input token lists by length
            all_input_seqs.extend(np.array(input_tokens)[order.cpu().numpy()].tolist())
            
            output_log_probs, output_seqs = model(input_variable, list(sorted_lengths))
            all_output_seqs.extend(trim_seqs(output_seqs))
            all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

            flattened_log_probs = output_log_probs.view(batch_size * model.max_length, -1)
            batch_losses = model.citerion(flattened_log_probs, target_variable.contiguous().view(-1))

            val_loss.update(batch_losses[0], batch_size)
        
    bleu_score = corpus_bleu(all_target_seqs, all_output_seqs, smoothing_function=SmoothingFunction().method1)
    
    logger.info('dev valid : Epoch = %d | Loss = %.2f | Bleu = %.2f' %
                (global_stats['epoch'], val_loss.avg * 100, bleu_score * 100) +
                '| examples = %d | valid time = %.2f (s)' %
                (len(all_output_seqs), eval_time.time()))
    
    if args.display_samples:
        for sentence_input, sentence_pred, sentence_gold in zip(all_input_seqs[-5:], all_output_seqs[-5:], all_target_seqs[-5:]):
            sentence_gold = sentence_gold[0]
            
            sentence_gold = seq_to_string(np.array(sentence_gold), vocab.id2word, input_tokens=sentence_input.split(' '))
            sentence_pred = seq_to_string(np.array(sentence_pred), vocab.id2word, input_tokens=sentence_input.split(' '))
                
            print('Predicted : %s ' % (sentence_pred))
            print('-----------------------------------------------')
            print('Gold : %s ' % (sentence_gold))
            print('===============================================')

    
    return {'bleu_score': bleu_score * 100}
    

def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = load_data(args.train_file, add_sql_symbol=False, add_query_symbol=True)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = load_data(args.dev_file, add_sql_symbol=False, add_query_symbol=True)
    logger.info('Num dev examples = %d' % len(dev_exs))
    
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    
    if args.checkpoint and os.path.isfile(args.model_file):
        pass
    else:
        logger.info('Training model from scratch...')
        model, vocab = init_from_scratch(args, train_exs, dev_exs)
        
        model.init_optimizer()
    
    if args.cuda:
        model.cuda()
        
    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = SequencePairDataset(
        examples=train_exs, 
        vocab=vocab, 
        args=args)
    dev_dataset = SequencePairDataset(
        examples=dev_exs, 
        vocab=vocab, 
        args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size)
    
    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}
    model.updates = 0
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)
        
        result = validate(args, dev_loader, model, vocab, stats)
        
        if result['bleu_score'] > stats['best_valid']:
            logger.info('Best valid: bleu score = %.2f (epoch %d, %d updates)' %
                        (result['bleu_score'],
                         stats['epoch'], model.updates))
            
            torch.save(model, args.model_file)
            stats['best_valid'] = result['bleu_score']
        
    
if __name__ == '__main__':
    # Parse cmdline args and setup enviroment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    args = parser.parse_args()
    set_defaults(args)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        
    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        
    # Set logging
        # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)