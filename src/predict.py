import os
import sys
import time
import torch
import argparse
import logging
import json
import jsonlines
import string

import numpy as np

from tqdm import tqdm
from utils import *
from data import SequencePairDataset

from torch.utils.data import DataLoader

import spacy

nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

ROOT_DIR = "./data"
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

parser = argparse.ArgumentParser(description='Parse predicting parameters')
parser.register('type', 'bool', str2bool)
parser.add_argument('--test-file', type=str,
                    default='train_processed.jsonl',
                    help='Preprocessed train file')

parser.add_argument('--model-name', type=str,
                    help='The name of a subdirectory of ./model/checkpoints/logs that '
                         'contains encoder and decoder model files.')

parser.add_argument('--out-dir', type=str, default='models/checkpoints/logs',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))

parser.add_argument('--batch-size', type=int, default=100,
                    help='The batch size to use when evaluating on the full dataset.')

parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')

parser.add_argument('--display-samples', type='bool', default=True,
                    help='Display top 5 samples in prediction process.')

parser.add_argument('--analysis-samples', type=int, default=100,
                    help='Sample n samples for analysis if not 0')

parser.add_argument('--use-extended-vocab', type='bool', default=True,
                    help='Use extended vocab for copying')

args = parser.parse_args()
t0 = time.time()

if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    args.cuda = False
    logger.info('Running on CPU only.')
    
# Defaults
ROOT_DIR = "./data"
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')
MODEL_DIR = './models/checkpoints/logs'
EMBED_DIR = os.path.join(ROOT_DIR, 'embeddings')

def predict(model, data_loader):
    """ Fetch the decoding probs of queries given sqls """
    all_output_seqs = []
    all_target_seqs = []
    all_query_probs = []
    all_input_seqs = []
    
    logger.info("Predicting")
    for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in tqdm(enumerate(data_loader)):
        lengths = (input_idxs != 0).long().sum(dim=1)
        sorted_lengths, order = torch.sort(lengths, descending=True)
        
        with torch.no_grad():
            input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
            target_variable = Variable(target_idxs[order, :])
            batch_size = input_variable.shape[0]
            all_input_seqs.extend(np.array(input_tokens)[order.cpu().numpy()].tolist())
            
            output_log_probs, output_seqs, query_probs = model.get_query_prob(input_variable, list(sorted_lengths), target_variable)
            
            trimmed_seqs, trimmed_probs = trim_seq_probs(output_seqs, query_probs)
            all_output_seqs.extend(trimmed_seqs)
            all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))
            all_query_probs.extend(trimmed_probs)
        
    all_confused_spans = []
    for output_seq, output_prob in zip(all_output_seqs, all_query_probs):
        assert len(output_seq) == len(output_prob)
        prev_prob = output_prob[1]
        span = []
        for idx, prob in zip(output_seq[2:], output_prob[2:]):
            # Confused span extraction strategy
            if prob < prev_prob * 20:
                span.append(idx)
            prev_prob = prob
                
        all_confused_spans.append(span)
    
    vocab = model.vocab
    results = []
    for output_seq, target_seq, output_prob, input_seq, confused_span in zip(all_output_seqs, all_target_seqs, all_query_probs, all_input_seqs, all_confused_spans):
        if len(confused_span) == 0:
            continue
        output_sentence = seq_to_string(np.array(output_seq), vocab.id2word, input_tokens=input_seq.split(' '))
        target_sentence = seq_to_string(np.array(target_seq[0]), vocab.id2word, input_tokens=input_seq.split(' '))
        confused_span = seq_to_string(np.array(confused_span), vocab.id2word, input_tokens=input_seq.split(' '))
        
        # Remove stop_words and punctuation
        confused_span = [t for t in confused_span.split(' ') 
                         if not (t in stop_words or string.punctuation.find(t) != -1 or t == '<s>' or t == '</s>')]
        if len(confused_span) == 0:
            continue
        confused_span = ' '.join(confused_span)
        
        r = {"output_sentence": output_sentence,
             "target_sentence": target_sentence,
             "confused_span": confused_span,
             "output_probs": ' '.join([str(p) for p in output_prob]),
             "input_sentence": input_seq}
        results.append(r)
        
    modelname = os.path.splitext(os.path.basename(args.model_name))[0]
    outfile = os.path.join(args.out_dir, modelname + '.preds')
    
    logger.info('Writing results to %s' % outfile)
    with jsonlines.open(outfile, 'w') as f:
        f.write_all(results)
            
    if args.display_samples:
        for r in np.random.choice(results, 10, replace=False):
            print('Input : %s ' % (r['input_sentence']))
            print('-----------------------------------------------')
            print('Predicted : %s ' % (r['output_sentence']))
            print('-----------------------------------------------')
            print('Gold : %s ' % (r['target_sentence']))
            print('-----------------------------------------------')
            print('Span : %s ' % (r['confused_span']))
            print('-----------------------------------------------')
            print('Prob : %s ' % (r['output_probs']))
            print('===============================================')
            
    if args.analysis_samples != 0:
        f = open(os.path.join(args.out_dir, modelname + '.analysis'), 'w')
        for r in np.random.choice(results, args.analysis_samples, replace=False):
            f.write('Input : %s \n' % (r['input_sentence']))
            f.write('-----------------------------------------------\n')
            f.write('Predicted : %s \n' % (r['output_sentence']))
            f.write('-----------------------------------------------\n')
            f.write('Gold : %s \n' % (r['target_sentence']))
            f.write('-----------------------------------------------\n')
            f.write('Span : %s \n' % (r['confused_span']))
            f.write('-----------------------------------------------\n')
            f.write('Prob : %s \n' % (r['output_probs']))
            f.write('===============================================\n')
        

def main(args):
    model_path = os.path.join(MODEL_DIR, args.model_name)
    if args.cuda:
        model = torch.load(model_path + '.pt')
    else:
        model = torch.load(model_path + '.pt', map_location=lambda storage, loc: storage)
        
    if args.cuda:
        model = model.cuda()
    else:
        model = model.cuda()
    
    args.test_file = os.path.join(DATA_DIR, args.test_file)
    
    logger.info('-' * 100)
    logger.info('Load test files')
    test_exs = load_data(args.test_file, add_sql_symbol=False, add_query_symbol=True)
    logger.info('Num test examples = %d' % len(test_exs))
    
    logger.info('-' * 100)
    logger.info('Make data loaders')
    
    vocab = model.vocab
    args.max_length = model.max_length
    
    dataset = SequencePairDataset(
        examples=test_exs, 
        vocab=vocab, 
        args=args)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    
    predict(model, data_loader)
        
    logger.info('Total time: %.2f' % (time.time() - t0))
    
    
if __name__ == '__main__':
    main(args)
    