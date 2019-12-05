import torch
dict_f = "dict.pt"
dict_checkpoint = torch.load(dict_f)

input_lang = dict_checkpoint["input_lang"]
output_lang = dict_checkpoint["output_lang"]

input_lang_dev = dict_checkpoint["input_lang_dev"]
output_lang_dev = dict_checkpoint["output_lang_dev"]

input_lang_devtest = dict_checkpoint["input_lang_devtest"]
output_lang_devtest = dict_checkpoint["output_lang_devtest"]

pairs = dict_checkpoint["pairs"]
pairs_dev = dict_checkpoint["pairs_dev"]
pairs_devtest = dict_checkpoint["pairs_devtest"]

MAX_LENGTH = 50
PAD_token = 0
SOS_token = 1
EOS_token = 2

max_epoch = 30
batch_size = 96
text_size = len(pairs)
hidden_size = 512
drop_out = 0.3
