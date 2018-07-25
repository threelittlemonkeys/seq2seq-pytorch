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
        pe = pos_encoder(EMBED_SIZE)
        self.encoder = encoder(vocab_size, pe)
        self.decoder = None

        if CUDA:
            self = self.cuda()

    def forward(self, x, mask):
        self.encoder(x, mask)

class encoder(nn.Module):
    def __init__(self, vocab_size, pe):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pe
        self.layers = nn.ModuleList([encoder_layer() for _ in range(NUM_LAYERS)])

    def forward(self, x, mask):
        x = self.embed(x)
        x += self.pe(x.size(1))
        for layer in self.layers:
            x = layer(x, mask)
            print(x.size())

class decoder(nn.Module):
    def __init__(self, vocab_size, pe):
        super().__init__()

        # architecture
        self.pe = pe

    def forward(self, x, mask):
        return

class encoder_layer(nn.Module):
    def __init__(self):
        super().__init__()

        # architecture
        self.attn = attn_mh(8, EMBED_SIZE // 8)
        self.ffn = ffn(2048)
        self.dropout = nn.Dropout(DROPOUT)
        self.res = lambda x, z: x + self.dropout(z) # residual connection and dropout
        self.norm = nn.LayerNorm(EMBED_SIZE) # layer normalization

    def forward(self, x, mask):
        z = self.attn(x, mask)
        z = self.norm(self.res(x, z))
        z = self.ffn(z)
        z = self.norm(self.res(x, z))
        return z

class pos_encoder(nn.Module): # positional encoding
    def __init__(self, d, maxlen = 1000):
        super().__init__()
        self.pe = Tensor(maxlen, d)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        k = torch.exp(-math.log(10000) * torch.arange(0, d, 2) / d)
        self.pe[:, 0::2] = torch.sin(pos * k)
        self.pe[:, 1::2] = torch.cos(pos * k)

    def forward(self, n):
        return self.pe[:n]

class attn_mh(nn.Module): # multi-head self-attention
    def __init__(self, h, d):
        super().__init__()
        self.h = h # number of heads
        self.d = d # dimension of each head

        # architecture
        self.Wq = nn.Linear(EMBED_SIZE, self.h * self.d)
        self.Wk = nn.Linear(EMBED_SIZE, self.h * self.d)
        self.Wv = nn.Linear(EMBED_SIZE, self.h * self.d)
        self.Wo = nn.Linear(self.h * self.d, EMBED_SIZE)

    def attn_sdp(self, q, k, v, mask): # scaled dot-product attention
        a = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d)
        mask = mask[0].unsqueeze(1).unsqueeze(3).expand_as(a)
        a = a.masked_fill(1 - mask, -10000) # masking in log space
        a = F.softmax(a, 2)
        a = torch.matmul(a, v)
        return a # attention weights

    def forward(self, x, mask):
        q = self.Wq(x).view(BATCH_SIZE, -1, self.h, self.d).transpose(1, 2)
        k = self.Wk(x).view(BATCH_SIZE, -1, self.h, self.d).transpose(1, 2)
        v = self.Wv(x).view(BATCH_SIZE, -1, self.h, self.d).transpose(1, 2)
        z = self.attn_sdp(q, k, v, mask)
        z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, self.h * self.d)
        z = self.Wo(z)
        return z

class ffn(nn.Module): # position-wise feed-forward networks
    def __init__(self, d):
        super().__init__()

        # architecture
        self.layers = nn.Sequential(
            nn.Linear(EMBED_SIZE, d),
            nn.ReLU(),
            nn.Linear(d, EMBED_SIZE),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.layers(x)

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
