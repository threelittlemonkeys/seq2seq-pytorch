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

    def forward(self, x, mask):
        self.encoder(x, mask)

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.layers = nn.ModuleList([encoder_layer() for _ in range(NUM_LAYERS)])

    def forward(self, x, mask):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
            print(x.size())

class encoder_layer(nn.Module):
    def __init__(self):
        super().__init__()

        # architecture
        self.attn = attn()
        self.ffn = ffn()
        self.norm = layer_norm(EMBED_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def res_block(self, x, z): # residual connection
        z = self.dropout(z) # residual dropout
        return self.norm(x + z)

    def forward(self, x, mask):
        z = self.res_block(x, self.attn(x, mask))
        z = self.res_block(z, self.ffn(z))
        return z

class attn(nn.Module): # multi-head self-attention
    def __init__(self):
        super().__init__()
        self.h = 8 # number of heads
        self.d = EMBED_SIZE // self.h # dimension of each head

        # architecture
        self.Wq = nn.Linear(EMBED_SIZE, self.h * self.d)
        self.Wk = nn.Linear(EMBED_SIZE, self.h * self.d)
        self.Wv = nn.Linear(EMBED_SIZE, self.h * self.d)
        self.Wo = nn.Linear(self.h * self.d, EMBED_SIZE)

    def sdp(self, q, k, v, mask): # scaled dot-product attention
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d)
        mask = mask[0].unsqueeze(1).unsqueeze(3).expand_as(score)
        score = score.masked_fill(1 - mask, -10000) # masking in log space
        score = F.softmax(score, 2)
        score = torch.matmul(score, v)
        return score

    def forward(self, x, mask):
        q = self.Wq(x).view(BATCH_SIZE, -1, self.h, self.d).transpose(1, 2)
        k = self.Wk(x).view(BATCH_SIZE, -1, self.h, self.d).transpose(1, 2)
        v = self.Wv(x).view(BATCH_SIZE, -1, self.h, self.d).transpose(1, 2)
        score = self.sdp(q, k, v, mask)
        score = score.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, self.h * self.d)
        score = self.Wo(score)
        return score

class ffn(nn.Module): # position-wise feed-forward networks
    def __init__(self):
        super().__init__()
        d = 2048

        # architecture
        self.layers = nn.Sequential(
            nn.Linear(EMBED_SIZE, d),
            nn.ReLU(),
            nn.Linear(d, EMBED_SIZE),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        y = self.layers(x)
        return y

class layer_norm(nn.Module): # layer normalization (Ba et al 2016)
    def __init__(self, H, eps = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(H))
        self.b = nn.Parameter(torch.zeros(H))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.g * (x - mean) / (std + self.eps) + self.b

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
