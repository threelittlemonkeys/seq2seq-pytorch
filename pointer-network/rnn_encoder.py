from utils import *
from embedding import *

class rnn_encoder(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()

        # architecture
        self.embed = embed(EMBED, cti, wti, batch_first = True, hre = HRE)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

    def init_state(self, b): # initialize states

        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "GRU":
            return hs
        cs = zeros(n, b, h) # LSTM cell state
        return (hs, cs)

    def forward(self, xc, xw, lens):

        b = len(lens)
        s = self.init_state(b)

        h = xh = self.embed(b, xc, xw)
        h = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first = True, enforce_sorted = False)
        h, s = self.rnn(h, s)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        return xh, h, s
