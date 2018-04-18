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
VERBOSE = True
SAVE_EVERY = 10

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()
CUDA = False

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

    def forward(self, x):
        lens = [len(seq) for seq in x]
        self.hidden = self.init_hidden("GRU")
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
        y, _ = self.rnn(x, self.hidden)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first = True)
        return y

class decoder(nn.Module): # vanilla
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
        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, x):
        lens = [len(seq) for seq in x]
        x = self.embed(x)
        h, _ = self.rnn(x, self.hidden)
        y = self.out(h).squeeze(1)
        y = self.softmax(y)
        return y

class decoder_attn(nn.Module): # attention-based
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
        self.attn = attention()
        self.cat = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        self.out = nn.Linear(HIDDEN_SIZE, vocab_size)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, x, enc_out):
        x = self.embed(x)
        h, _ = self.rnn(x, self.hidden)
        c = self.attn(h, enc_out)
        h = torch.cat((h.squeeze(1), c.squeeze(1)), 1)
        h = F.tanh(self.cat(h)) # attentional vector
        y = self.out(h)
        y = self.softmax(y)
        return y

class attention(nn.Module): # attention layer (Luong 2015)
    def __init__(self):
        super().__init__()
        self.fnn = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def score(self, h_tgt, h_src):
        batch_len = h_src.size(1)
        score = Var(zeros(BATCH_SIZE, batch_len))
        for b in range(BATCH_SIZE):
            ht = h_tgt[b, 0] # current target hidden state
            for t in range(batch_len):
                hs = h_src[b, t] # encoder output
                score[b, t] = ht.dot(self.fnn(hs)) # general product
        return score

    def forward(self, h_tgt, h_src):
        a = F.softmax(self.score(h_tgt, h_src), dim = 1) # alignment weight vector
        c = a.unsqueeze(1).bmm(h_src) # context vector
        return c

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

def len_unpadded(x): # get unpadded sequence length
    return next((i for i, j in enumerate(x) if scalar(j) == 0), len(x))
