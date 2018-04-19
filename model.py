import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

BATCH_SIZE = 64
EMBED_SIZE = 100
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
VERBOSE = False
SAVE_EVERY = 10

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn = nn.GRU( # LSTM or GRU
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )

        if CUDA:
            self = self.cuda()

    def init_hidden(self, rnn_type): # initialize hidden states
        h = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden states
        if rnn_type == "LSTM":
            c = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell states
            return (Var(h), Var(c))
        return Var(h)

    def forward(self, x, x_mask):
        lens = [sum(seq) for seq in x_mask]
        self.hidden = self.init_hidden("GRU") # LSTM or GRU
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
        y, _ = self.rnn(x, self.hidden)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first = True)
        return y

class decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.attn = True

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn = nn.GRU( # LSTM or GRU
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )
        if self.attn: # global attention (Luong 2015)
            self.Wa = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(-1)

        if CUDA:
            self = self.cuda()

    def forward(self, dec_in, enc_out = None, x_mask = None):
        dec_in = self.embed(dec_in)
        h, _ = self.rnn(dec_in, self.hidden)
        if self.attn:
            a = h.bmm(self.Wa(enc_out).transpose(1, 2)) # alignment scores
            a.masked_fill_(Var(1 - x_mask.unsqueeze(1)), -10000)
            a = F.softmax(a, dim = -1) # alignment vector
            c = a.bmm(enc_out) # context vector
            h = torch.cat((h, c), -1)
            h = F.tanh(self.Wc(h)) # attentional vector
        y = self.out(h).squeeze(1)
        y = self.softmax(y)
        return y

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]
