from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "SOS": 1, "EOS": 2, "<UNK>": 3}
        #self.word2index = {}
        self.word2count = {"SOS": 2, "EOS": 2, "<PAD>": 2, "<UNK>": 2}
        self.index2word = {0: "<PAD>", 1: "SOS", 2: "EOS", 3: "<UNK>"}
        self.n_words = 4  # Count SOS and EOS and PAD and UNK
    
    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)
            
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def rmWord(self):
        for k, v in list(self.index2word.items()):
            if self.word2count.get(v) == 1:
                del self.word2count[v], self.word2index[v], self.index2word[k]
                self.n_words -= 1
        
        cnt = 0
        for k, v in list(self.index2word.items()):
            self.index2word[cnt] = v
            self.word2index[v] = cnt
            cnt += 1
    

def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"[[０-９．]+]", r"", s).strip()
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

TRAIN_PATH = "../corpus.tok/train-1.%s.tok.txt"
DEV_PATH = "../corpus.tok/dev.%s.tok.txt"
DEVTEST_PATH = "../corpus.tok/devtest.%s.tok.txt"

def readLangs(lang1, lang2, flag=0, reverse=False):
    print("Reading lines...")
    
    # Read the file and split into lines
    if flag == 0:
        print("train")
        lines1 = open(TRAIN_PATH % (lang1), encoding='utf-8').read().strip().split('\n')
        #lines1 = open('./corpus.tok/v7_train-1.%s.tok.txt' % (lang1), encoding='utf-8').read().strip().split('\n')
        
        lines2 = open(TRAIN_PATH % (lang2), encoding='utf-8').read().strip().split('\n')
        #lines2 = open('./corpus.tok/v7_train-1.%s.tok.txt' % (lang2), encoding='utf-8').read().strip().split('\n')
    elif flag == 1:
        print("dev")
        lines1 = open(DEV_PATH % (lang1), encoding='utf-8').read().strip().split('\n')
        #lines1 = open('./corpus.tok/v7_dev.%s.tok.txt' % (lang1), encoding='utf-8').read().strip().split('\n')
        
        lines2 = open(DEV_PATH % (lang2), encoding='utf-8').read().strip().split('\n')
    else :
        print("devtest")
        lines1 = open(DEVTEST_PATH % (lang1), encoding='utf-8').read().strip().split('\n')
        #lines1 = open('./corpus.tok/v7_devtest.%s.tok.txt' % (lang1), encoding='utf-8').read().strip().split('\n')
        
        lines2 = open(DEVTEST_PATH % (lang2), encoding='utf-8').read().strip().split('\n')
    
    
    # Split every line into pairs and normalize
    pairs = []
    for idx in range(len(lines1)):
        list1 = lines1[idx].split(" ")
        while "" in list1:
            list1.remove("")
        list1 = " ".join(list1)
        list1 = normalizeString(list1)
        
        list2 = lines2[idx].split(" ")
        while "" in list2:
            list2.remove("")
        while "　" in list2:
            list2.remove("　")
        
        list2 = " ".join(list2)
        #list2 = normalizeString(list2)
        
        pairs.append([list1,list2])
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 50

def filterPair(p):
    return len(p[0].split(' ')) <= MAX_LENGTH and len(p[1].split(' ')) <= MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, flag=0, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, flag, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if flag == 0:
        pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    input_lang.rmWord()    # cut dict size
    output_lang.rmWord()   # cut dict size
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
    
def main():
    flag = 2
    input_lang_devtest, output_lang_devtest, pairs_devtest = prepareData('en', 'ja', flag, False)
    flag = 1
    input_lang_dev, output_lang_dev, pairs_dev = prepareData('en', 'ja', flag, False)
    flag = 0
    input_lang, output_lang, pairs = prepareData('en', 'ja', flag, False)
    
    f_name = "dict.pt"
    torch.save({
        "input_lang": input_lang,
        "output_lang": output_lang,
        "pairs": pairs,
        "input_lang_dev": input_lang_dev,
        "output_lang_dev": output_lang_dev,
        "pairs_dev": pairs_dev,
        "input_lang_devtest": input_lang_devtest,
        "output_lang_devtest": output_lang_devtest,
        "pairs_devtest": pairs_devtest,
    }, f_name)


if __name__ == "__main__":
    main()
