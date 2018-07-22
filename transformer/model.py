import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 128
EMBED_SIZE = 500
HIDDEN_SIZE = 1000
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

class transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = encoder(vocab_size)
        self.decoder = None

        if CUDA:
            self = self.cuda()

    def forward(self, x, mask):
        self.encoder(x)

class encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.layers = nn.Sequential(*[enc_layer(vocab_size) for _ in range(NUM_LAYERS)])

    def forward(self, x):
        print(x.size())
        x = self.embed(x)
        print(x.size())
        h = self.layers(x)
        exit()

class enc_layer(nn.Module): # encoder layer
    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.attn = attn(vocab_size)

    def forward(self, x):
        self.attn(x)
        return x

class attn(nn.Module): # multi-head attention
    def __init__(self, vocab_size):
        super().__init__()

    def forward(self, x):
        print(x.size())
        return x

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
