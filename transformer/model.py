import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 128
EMBED_SIZE = 512 # representation dimension
NUM_LAYERS = 6
DROPOUT = 0.5
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
CUDA = False

class transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = encoder(vocab_size)
        self.decoder = None

        if CUDA:
            self = self.cuda()

    def forward(self, x, mask = None):
        self.encoder(x, mask)

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.layers = nn.ModuleList([enc_layer(vocab_size) for _ in range(NUM_LAYERS)])

    def forward(self, x, mask):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)

class enc_layer(nn.Module): # encoder layer
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.attn = attn(vocab_size)

    def forward(self, x, mask):
        self.attn(x, mask)

class attn(nn.Module): # multi-head attention
    def __init__(self, vocab_size):
        super().__init__()
        self.h = 8 # number of heads
        self.d = EMBED_SIZE // self.h # dimension of each head

        # architecture
        self.Wq = nn.Parameter(Tensor(self.h, EMBED_SIZE, self.d))
        self.Wk = nn.Parameter(Tensor(self.h, EMBED_SIZE, self.d))
        self.Wv = nn.Parameter(Tensor(self.h, EMBED_SIZE, self.d))

    def forward(self, x, mask):
        h = x.unsqueeze(1)
        q = torch.matmul(h, self.Wq) # query
        k = torch.matmul(h, self.Wk) # key 
        v = torch.matmul(h, self.Wv) # value
        # scaled dot-product attention
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d)
        # mask
        score = F.softmax(score, 2)
        score = torch.matmul(score, v)
        score = score.transpose(1, 2).contiguous().view_as(x)
        print(score.size())
        exit()

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def maskset(x):
    mask = x.data.gt(0)
    return (mask, [sum(seq) for seq in mask]) # set of mask and lengths
