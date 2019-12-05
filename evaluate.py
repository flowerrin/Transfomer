from __future__ import unicode_literals, print_function, division
from tqdm import tqdm
from nltk import bleu_score
import unicodedata
import string
import re
import random
import argparse
import os
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformer import Transformer
from trans_encoder import TransEncoder
from trans_decoder import TransDecoder
from mydataset import MyDataset
from prepare import Lang

from datetime import datetime


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
    if flag == 0:
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


def tensorsFromPair(input_b, output_b):
    input_tensor = tensorFromSentence(input_lang, input_b)
    target_tensor = tensorFromSentence(output_lang, output_b)
    return (input_tensor, target_tensor)


def evaluate(transformer, sentence, batch_size=batch_size, flag=1):
    #model_name = "seq2seq_model_v{}.pt".format(max_epoch)
    model_name = "./models/seq2seq_model.pt"
    #model_name = "seq2seq_model_max.pt"
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
            decoder_output = transformer.decode(decoder_input, encoder_output, input_tensor, 0)  # [batch, 1, vocab_size]
            index = torch.argmax(decoder_output[:, -1:].cpu().detach(), dim=-1).to(device)  # [batch, 1]
            
            eq = torch.eq(index[:, -1:].clone(), PAD_tokens)
            if eq.sum == batch_size:
                break
            else:
                pass
            
            decoder_input = torch.cat([decoder_input, index[:, -1:]], dim=-1)  # [batch, 2---length]
            
        decoded_words = decoder_input[:, 1:]
        
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
    data_w = open("result_eva.txt", "w")
    
    pair = pairs_devtest
    inputs, outputs = map(list, zip(*pair))
    test_pairs = [tensorsFromPair(inputs[i].split(" "), outputs[i].split(" "))
                  for i in range(n)]
    
    dataset = MyDataset(test_pairs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0, collate_fn=my_collate_fn)
    
    # idx : [inputs, targets], [ip_len, op_len]
    for idx in tqdm(dataloader, ascii = True):
        batch = len(idx[0][0])
        test_pair = idx[0]
        input_tensor = torch.tensor(test_pair[0], dtype=torch.long, device=device)
        
        output_words = evaluate(transformer, input_tensor, batch)
        
        for s in output_words:
            if len(s) != 0:
                data_w.write(s+"\n")
            else:
                data_w.write("\n")
    
    data_w.close()


def main():
    trans_encoder1 = TransEncoder(input_lang.n_words, hidden_size, batch_size, drop_out).to(device)
    trans_decoder1 = TransDecoder(hidden_size, output_lang.n_words, batch_size, drop_out).to(device)
    
    transformer = Transformer(trans_encoder1, trans_decoder1).to(device)
    
    result_write(transformer, len(pairs_devtest))

if __name__ == "__main__":
    main()
