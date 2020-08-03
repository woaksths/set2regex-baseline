#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import torch
import torchtext
import argparse
import time
import seq2seq
import subprocess
import re


from regexDFAEquals import regex_equiv_from_raw, unprocess_regex, regex_equiv
from seq2seq.optim import Optimizer
from seq2seq.models import EncoderRNN, DecoderRNN,Seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.string_preprocess import get_set_num,pad_tensor, count_star, decode_tensor_input, decode_tensor_target
from seq2seq.util.regex_operation import pos_membership_test, neg_membership_test, preprocess_regex, regex_equal,regex_inclusion,valid_regex, or_exception



parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', help='path to train data')
parser.add_argument('--test_path', action='store', dest='test_path', help='path to test data')
parser.add_argument('--checkpoint', action='store', dest='checkpoint', help='path to checkpoint')
opt = parser.parse_args()

latest_check_point = Checkpoint.get_latest_checkpoint(opt.checkpoint)
checkpoint = Checkpoint.load(latest_check_point)
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab
print(latest_check_point)
print(input_vocab.stoi)
print(output_vocab.stoi)

model = checkpoint.model
optimizer = checkpoint.optimizer
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi['<pad>']
loss = NLLLoss(weight, pad)
batch_size = 1 
print(model)

train_file = opt.train_path
test_file = opt.test_path
src = SourceField()
tgt = TargetField()

train = torchtext.data.TabularDataset(
    path=train_file, format='tsv',
    fields=[('src', src), ('tgt', tgt)]
)
test_data = torchtext.data.TabularDataset(
    path=test_file, format='tsv',
    fields=[('src', src), ('tgt', tgt)]
)

src.build_vocab(train, max_size=500)
tgt.build_vocab(train, max_size=500)

device = torch.device('cuda:0') if torch.cuda.is_available() else -1
batch_iterator = torchtext.data.BucketIterator(
    dataset=test_data, batch_size=1,
    sort=False, sort_within_batch=False,sort_key=lambda x: len(x.src),
    device=device, repeat=False, shuffle=False, train=False)

model.eval()
loss.reset()
start = time.time()

match = 0
total = 0
num_samples = 0

with torch.no_grad():
    with open('{}_error_analysis.txt'.format(opt.checkpoint), 'w') as fw:
        statistics = [{'cnt':0,'hit': 0,'string_equal':0,'dfa_equal':0, 'membership_equal':0, 'invalid_regex':0} for _ in range(4)]
        for batch in batch_iterator:
            num_samples = num_samples + 1
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)
            
            with torch.no_grad():
                softmax_list, _, other =model(input_variables, input_lengths)
            
            length = other['length'][0]
            tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
            tgt_seq = [output_vocab.itos[tok] for tok in tgt_id_seq]
            
            # calculate NLL accuracy
            non_padding = target_variables.view(-1)[1:].ne(pad).type(torch.LongTensor)
            predict_var = torch.stack(tgt_id_seq)
            target_var = target_variables.view(-1)[1:]

            max_len = max(len(predict_var), len(target_var))
            padded_predict_var = pad_tensor(predict_var, max_len, output_vocab)
            padded_target_var = pad_tensor(target_var, max_len, output_vocab)
            padded_non_padding = pad_tensor(non_padding, max_len, output_vocab).type(torch.bool)
            correct = padded_predict_var.eq(padded_target_var).masked_select(padded_non_padding).sum().item()

            match += correct
            total += non_padding.sum().item()

            predict_regex = ' '.join(tgt_seq[:-1])
            target_regex = decode_tensor_target(target_variables, output_vocab)
            target_regex, predict_regex = preprocess_regex(target_regex, predict_regex)
                        
            for i in range(input_variables.size(0)):
                res = ""
                for j in range(input_variables[i].size(0)):
                    res += input_vocab.itos[input_variables[i][j]]
                res = res.replace('<pad>', '')
            
            inputs = res.split('<sep>')
            pos_input = inputs[:int(len(inputs)/2)]
            neg_input = inputs[int(len(inputs)/2):]
            star_cnt = count_star(target_regex)

            statistics[star_cnt]['cnt'] +=1            
            # calculate regex equivalent accuracy
            
            if not valid_regex(predict_regex) or not or_exception(predict_regex):
                statistics[star_cnt]['invalid_regex'] +=1
            else:    
                if target_regex == predict_regex:
                    statistics[star_cnt]['hit'] +=1
                    statistics[star_cnt]['string_equal']+=1
                elif regex_equal(target_regex, predict_regex):
                    statistics[star_cnt]['hit'] +=1
                    statistics[star_cnt]['dfa_equal'] +=1
                elif pos_membership_test(predict_regex, pos_input) and neg_membership_test(predict_regex,neg_input):
                    statistics[star_cnt]['hit'] +=1 
                    statistics[star_cnt]['membership_equal'] +=1
                else: 
                    fw.write('pos_input : ' + ' '.join(pos_input)+'\n')
                    fw.write('neg_input : ' + ' '.join(neg_input)+'\n')
                    fw.write('target_regex : ' + target_regex  +'\n')
                    fw.write('predict_regex : ' + predict_regex + '\n\n')
            
            if total == 0:
                accuracy = float('nan')
            else:
                accuracy = match / total
                
            if num_samples % 100 == 0:
                print("Iterations: ", num_samples, ", " , "{0:.3f}".format(accuracy) + '\n')
            
end = time.time()
print(statistics)
print(end-start)