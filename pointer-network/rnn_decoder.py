from utils import *
from embedding import *

class rnn_decoder(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()
        self.hs = None # source hidden state
        self.hidden = None # decoder hidden state

        self.M = None # encoder hidden states
        self.H = None # decoder hidden states
        self.h = None # decoder output

        # architecture
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = sum(EMBED.values()),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()

    def forward(self, yi, mask):

        h, self.H = self.rnn(yi, self.H)
        yo = self.attn(self.M, h, mask)

        return yo

class attn(nn.Module): # content based input attention

    def __init__(self):

        super().__init__()

        # architecture
        self.W1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.W2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.V = nn.Linear(HIDDEN_SIZE, 1)
        self.W = None # attention weights

    def forward(self, hs, ht, mask):

        u = self.V(torch.tanh(self.W1(hs) + self.W2(ht))) # [B, L, H] -> [B, L, 1]
        u = u.squeeze(2).masked_fill(mask, -10000)
        self.W = F.log_softmax(u, 1) # [B, L]

        return self.W
