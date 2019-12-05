class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "SOS": 1, "EOS": 2, "<UNK>": 3}
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
            
