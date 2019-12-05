from __future__ import unicode_literals, print_function, division
from tqdm import tqdm
from nltk import bleu_score
from torch.autograd import Variable
import unicodedata
import string
import re
import random
import argparse
import os
import time
import math
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformer import Transformer
from trans_encoder import TransEncoder
from trans_decoder import TransDecoder
from mydataset import MyDataset
from labelsmoothing import LabelSmoothing
from prepare import Lang

from datetime import datetime

#import multiprocessing as mp
#mp.set_start_method('spawn')

import define
dict_f = define.dict_f
dict_checkpoint = torch.load(dict_f)

input_lang = define.input_lang
output_lang = define.output_lang
pairs = define.pairs

input_lang_dev = define.input_lang_dev
output_lang_dev = define.output_lang_dev
pairs_dev = define.pairs_dev

input_lang_devtest = define.input_lang_devtest
output_lang_devtest = define.output_lang_devtest
pairs_devtest = define.pairs_devtest

MAX_LENGTH = define.MAX_LENGTH
PAD_token = define.PAD_token
SOS_token = define.SOS_token
EOS_token = define.EOS_token

max_epoch = define.max_epoch
batch_size = define.batch_size
text_size = define.text_size
hidden_size = define.hidden_size
drop_out = define.drop_out


def padding(lang_list, max_len, flag=0):
    if flag == 1:
        lang_list.append(EOS_token)
    else:
        lang_list = lang_list[::-1]   # reverse
    
    sub = max_len - len(lang_list)
    for cnt in range(sub):
        lang_list.append(PAD_token)
    return lang_list


def my_collate_fn(batch):
    # datasetの出力
    # inputs = [batch, length]
    # [inputs, targets] = dataset[batch_idx]
    inputs = []
    targets = []
    
    lengths = [(len(i[0]), len(i[1])) for i in batch]
    max_len = list(map(max, zip(*lengths)))
    ip_len, op_len = map(list, zip(*lengths))
    op_len = map(lambda x: x + 1, op_len)
    
    pad_inputs = []
    pad_targets = []
    for sample in batch:
        input_s, target_s = sample
        pad_inputs = padding(input_s, max_len[0])
        pad_targets = padding(target_s, max_len[1]+1, 1)
        inputs.append(pad_inputs)
        targets.append(pad_targets)
        
    return [inputs, targets], [ip_len, op_len]


def indexesFromSentence(lang, sentence):
    langs = []
    word = sentence
    if word in lang.word2index:
        langs = lang.word2index[word]
    else:
        langs = lang.word2index["<UNK>"]
    return langs


def tensorFromSentence(lang, sentence, flag=0):
    indexes = []
    for idx in range(len(sentence)):
        indexes.append(indexesFromSentence(lang, sentence[idx]))
    return indexes


def tensorsFromPair(input_b, output_b, flag=0):
    i_lang = input_lang
    o_lang = output_lang
    input_tensor = tensorFromSentence(i_lang, input_b)
    target_tensor = tensorFromSentence(o_lang, output_b)
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, sq_lengths, transformer, transformer_optimizer, criterion, all_step_cnt, batch_size=batch_size):
    transformer_optimizer.zero_grad()
    
    input_length = input_tensor.shape[-1]
    target_length = target_tensor.shape[-1]
    batch_length = batch_size
    
    loss = 0
    
    tmp_tensor = target_tensor.clone()
    tmp_tensor[target_tensor==EOS_token] = PAD_token
    tmp_tensor = tmp_tensor[:, :-1]
    
    SOS_tokens = [[SOS_token]]
    decoder_input = torch.tensor(SOS_tokens, device=device)
    decoder_input = decoder_input.repeat(batch_size, 1)
    
    decoder_input = torch.cat([decoder_input, tmp_tensor], dim=1)
    
    decoder_output = transformer(input_tensor, decoder_input, 1)
    
    # Teacher forcing: Feed the target as the next input
    #  [batch, length, vocab_size] , [batch, length]    
    #decoder_output = decoder_output.reshape(-1, output_lang.n_words)
    #target_tensor = target_tensor.reshape(-1)
    loss += criterion(decoder_output, target_tensor)   # [batch, length, vocab_size], [batch, length]
    
    op_lengths = sq_lengths[1]
    token_sum = sum(op_lengths)
    
    loss = loss / token_sum
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=5.0)
    
    learning_rate = get_learning_rate(all_step_cnt, max_epoch)
    for p in transformer_optimizer.param_groups:
        p["lr"] = learning_rate
    transformer_optimizer.step()
    
    #transformer_optimizer.zero_grad()
    
    return loss.item()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, since_epo, percent):
    now = time.time()
    all_s = now - since
    s = now - since_epo
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(all_s), asMinutes(rs))


def get_learning_rate(step, max_epoch):
    warmup_step = 4000
    #warmup_step = 3000
    #warmup_step = int(40*max_epoch)
    
    rate = hidden_size ** (-0.5) * min(step ** (-0.5), step * warmup_step ** (-1.5))
    return rate

                                                               #0.0001, 0.001
def trainIters(transformer, n_iters, print_every, learning_rate=0):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    
    #transformer_optimizer = optim.SGD(transformer.parameters(), lr=learning_rate)
    transformer_optimizer = optim.Adam(transformer.parameters(), lr=learning_rate, weight_decay=1e-06, betas=(0.9, 0.98), eps=1e-09)
                                                                                   #weight_decay=1e-06 , 0.002
                                                                                   
    criterion = LabelSmoothing(PAD_token, 0.1)
    #criterion = nn.NLLLoss(ignore_index=PAD_token, reduction='sum')
    
    global max_epoch
    BLEUs = {}
    all_step_cnt = 1
    
    inputs, outputs = map(list, zip(*pairs))
    training_pairs = [tensorsFromPair(inputs[i].split(" "), outputs[i].split(" "))
                      for i in range(text_size)]
    
    dataset = MyDataset(training_pairs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=my_collate_fn)
    #num_workers=3
    
    for epoch in range(1, max_epoch+1):
        start_epo = time.time()
        print_loss_avg_total = 0
        iter = 1
        
        transformer.train()
        
        # idx : [inputs, targets], [ip_len, op_len]
        # 1iter : [batch_size, length]
        for idx in dataloader:
            batch = len(idx[0][0])
            training_pair = idx[0]
            sq_lengths = idx[1]
            
            input_tensor = torch.tensor(training_pair[0], dtype=torch.long, device=device)
            target_tensor = torch.tensor(training_pair[1], dtype=torch.long, device=device)
            
            loss = train(input_tensor, target_tensor, sq_lengths,
                         transformer, transformer_optimizer, criterion, all_step_cnt, batch)
            print_loss_total += loss
            
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_avg_total += print_loss_avg
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, start_epo, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
            
            all_step_cnt += 1
            iter += 1
            
        print("Epoch %d Loss: %.4f" % (epoch, print_loss_avg_total/10))
        
        model_name = "./models/seq2seq_model_v{}.pt".format(epoch)
        torch.save({
            "transformer_model": transformer.state_dict(),
        }, model_name)
        
        
        all_BLEUscore = Validation(transformer, len(pairs_dev), epoch-1)
        print("Epoch %d BLEU : %.4f" % (epoch, all_BLEUscore))
        
        # Early Stopping
        if epoch >= 6:
            BLEU_list = list(BLEUs.values())
            if (sum(BLEU_list) / len(BLEU_list)) >= all_BLEUscore:
                max_BLEU_epoch = max(BLEUs, key=BLEUs.get)
                print("Max Epoch:", max_BLEU_epoch)
                max_epoch = max_BLEU_epoch
                
                model_name = "./models/seq2seq_model.pt"
                torch.save({
                    "transformer_model": transformer.state_dict(),
                }, model_name)
                break
            else:
                min_BLEU_epoch = min(BLEUs, key=BLEUs.get)
                del BLEUs[min_BLEU_epoch]
        BLEUs[epoch] = all_BLEUscore
        
        
def Validation(transformer, n, epoch):
    all_BLEUscore = 0
    new_n = int(n/max_epoch)
    start = 0
    stop = n
    
    with torch.no_grad():
        pair = pairs_dev
        inputs, outputs = map(list, zip(*pair))
        dev_pairs = [tensorsFromPair(inputs[i].split(" "), outputs[i].split(" "))
                     for i in range(n)]
        dataset = MyDataset(dev_pairs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0, collate_fn=my_collate_fn)
        
        cnt = 0
        res = []
        hy = []
        chencherry = bleu_score.SmoothingFunction()
        for idx in tqdm(dataloader, ascii = True):
            batch = len(idx[0][0])
            dev_pair = idx[0]
            
            input_tensor = torch.tensor(dev_pair[0], dtype=torch.long, device=device)
            re_tensor = torch.tensor(dev_pair[1], dtype=torch.long, device=device)
            
            output_words = evaluate(transformer, input_tensor, batch, 0)
            #output_words = evaluate(transformer, input_tensor, batch, -1)  #debag
            
            for i in range(len(output_words)):
                re = outputs[cnt].split(" ")
                res.append([ re ])
                if len(output_words[i]) != 0:   # 空判定
                    hy.append(output_words[i].split(" "))
                else:
                    hy.append([""])
                
                cnt += 1
                
        #all_BLEUscore += bleu_score.corpus_bleu(res, hy, smoothing_function=chencherry.method4)  # smoothing_function=chencherry.method4
        all_BLEUscore += bleu_score.corpus_bleu(res, hy)
        all_BLEUscore *= 100
        
    return all_BLEUscore


def evaluate(transformer, sentence, batch_size=batch_size, flag=1):
    
    if flag == 1:
        model_name = "./models/seq2seq_model_v{}.pt".format(max_epoch)
        checkpoint = torch.load(model_name)
        transformer.load_state_dict(checkpoint["transformer_model"])
    elif flag == -1:   # debag
        model_name = "./models/seq2seq_model_v{}.pt".format(12)
        checkpoint = torch.load(model_name)
        transformer.load_state_dict(checkpoint["transformer_model"])
        
    transformer.eval()
    
    with torch.no_grad():
        #[batch_size, length]
        input_tensor = sentence
        _, input_length = input_tensor.shape
        
        SOS_tokens = [[SOS_token]]
        decoder_input = torch.tensor(SOS_tokens, device=device)
        decoder_input = decoder_input.repeat(batch_size, 1)
        
        PAD_tokens = [[PAD_token]]
        PAD_tokens = torch.tensor(PAD_tokens, device=device)
        PAD_tokens = PAD_tokens.repeat(batch_size, 1)
        
        decoded_sentences = []
        decoded_words = decoder_input.clone()
        #max_length = 80
        max_length = input_length+30
        
        encoder_output = transformer.encode(input_tensor, 1)
        
        # encoder_output : [batch, length, hidden]
        # decoder_input : [batch, 1]
        for i in range(max_length):
            decoder_output = transformer.decode(decoder_input, encoder_output, input_tensor, 0)  # [batch, 1---length, vocab_size]
            index = torch.argmax(decoder_output[:, -1:].cpu().detach(), dim=-1).to(device)  # [batch, 1]
            
            eq = torch.eq(index[:, -1:].clone(), PAD_tokens)
            if eq.sum == batch_size:
                break
            else:
                pass
            
            decoder_input = torch.cat([decoder_input, index[:, -1:]], dim=-1)  # [batch, 2---length]
            
        decoded_words = decoder_input[:, 1:]  # [batch, length]  delete SOS
        
        for i in range(decoded_words.shape[0]):
            decoded_sentences.append(SentenceFromTensor(output_lang, decoded_words[i]))
        
        return decoded_sentences


def SentenceFromTensor(lang, sentence):
    langs = []
    for word in sentence:
        word = word.item()
        if lang.index2word[word] == "EOS":
            break
        else:
            langs.append(lang.index2word[word])
        
    if not len(langs) == 0:
        langs = " ".join(langs)
    return langs


def result_write(transformer, n):
    data_w = open("result.txt", "w")
    
    pair = pairs_devtest
    inputs, outputs = map(list, zip(*pair))
    test_pairs = [tensorsFromPair(inputs[i].split(" "), outputs[i].split(" "))
                  for i in range(n)]
    
    dataset = MyDataset(test_pairs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0, collate_fn=my_collate_fn)
    
    # idx : [inputs, targets], [ip_len, op_len]
    # 1iter : [batch_size, length]
    # batch=96, text_size=1784
    for idx in tqdm(dataloader, ascii = True):
        batch = len(idx[0][0])
        test_pair = idx[0]
        lengths = idx[1]
        op_len = lengths[1]
        
        input_tensor = torch.tensor(test_pair[0], dtype=torch.long, device=device)
        
        output_words = evaluate(transformer, input_tensor, batch)
        
        for s in output_words:
            if len(s) != 0:
                data_w.write(s+"\n")
            else:
                data_w.write("\n")
        
    data_w.close()
    
    
def main():
    print("train")
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print("TextSize:", text_size)
    
    trans_encoder1 = TransEncoder(input_lang.n_words, hidden_size, batch_size, drop_out).to(device)
    trans_decoder1 = TransDecoder(hidden_size, output_lang.n_words, batch_size, drop_out).to(device)
    
    transformer = Transformer(trans_encoder1, trans_decoder1).to(device)
    
    n_iter = int(text_size/batch_size)
    print_every = int(n_iter/10)
    
    # Xavier の初期値 (一様分布)
    # Xavier の初期値 (正規分布)
    # He の初期値
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            #nn.init.xavier_normal_(p)  #gain=nn.init.calculate_gain('relu')
            #nn.init.kaiming_uniform_(p, nonlinearity='relu')
        
    trainIters(transformer, n_iter, print_every)
    
    result_write(transformer, len(pairs_devtest))

if __name__ == "__main__":
    main()
