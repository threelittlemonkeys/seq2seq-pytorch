from utils import *
from embedding import embed

class rnn_encoder_decoder(nn.Module):
    def __init__(self, x_cti_size, x_wti_size, y_wti_size):
        super().__init__()

        # architecture
        self.enc = encoder(x_cti_size, x_wti_size)
        self.dec = decoder(y_wti_size)
        self = self.cuda() if CUDA else self

    def forward(self, xc, xw, y0): # for training
        b = y0.size(0) # batch size
        loss = 0
        self.zero_grad()
        mask, lens = maskset(xw)
        self.dec.M, self.dec.h = self.enc(b, xc, xw, lens)
        self.dec.H = self.enc.H
        self.dec.attn.V = zeros(b, 1, HIDDEN_SIZE)
        if METHOD == "copy":
            self.dec.copy.V = zeros(b, 1, HIDDEN_SIZE)
        yi = LongTensor([SOS_IDX] * b)
        for t in range(y0.size(1)):
            yo = self.dec(yi.unsqueeze(1), mask)
            yi = y0[:, t] # teacher forcing
            loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX)
        loss /= y0.size(1) # divide by senquence length
        return loss

    def decode(self, x): # for inference
        pass

class encoder(nn.Module):
    def __init__(self, cti_size, wti_size):
        super().__init__()
        self.H = None # encoder hidden states

        # architecture
        self.embed = embed(ENC_EMBED, cti_size, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

    def init_state(self, b): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, b, xc, xw, lens):
        self.H = self.init_state(b)
        x = self.embed(xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
        h, s = self.rnn(x, self.H)
        s = s[RNN_TYPE == "LSTM"][-NUM_DIRS:] # final hidden state
        s = torch.cat([_ for _ in s], 1).view(b, 1, -1)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h, s

class decoder(nn.Module):
    def __init__(self, wti_size):
        super().__init__()
        self.M = None # encoder hidden states
        self.H = None # decoder hidden states
        self.h = None # decoder output

        # architecture
        self.embed = embed(DEC_EMBED, 0, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim + HIDDEN_SIZE * (1 + 0),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        if METHOD == "attn":
            self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        if METHOD == "copy":
            self.copy = copy()
        self.Wo = nn.Linear(HIDDEN_SIZE, wti_size)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, y1, mask):
        x = self.embed(None, y1)

        if METHOD == "attn":
            x = torch.cat((x, self.attn.V), 2) # input feeding
            h, _ = self.rnn(x, self.H)
            self.attn(h, self.M, mask)
            h = self.Wc(torch.cat((self.attn.V, h), 2)).tanh()
            h = self.Wo(h).squeeze(1)
            y = self.softmax(h)
            return y

        if METHOD == "copy":
            self.attn(self.h, self.M, mask)
            x = torch.cat((x, self.attn.V), 2)
            self.h, _ = self.rnn(x, self.H)
            g = self.Wo(self.h).squeeze(1) # generation scores
            c = self.copy(self.h, self.M, mask) # copy scores
            p = self.softmax(torch.cat([g, c], 1))
            g, c = p.split([g.size(1), c.size(1)], 1)
            # TODO
            return p

class attn(nn.Module): # attention mechanism
    def __init__(self):
        super().__init__()

        # architecture
        self.Wa = None # attention weights
        self.V = None # context vector

    def forward(self, ht, hs, mask):
        self.Wa = ht.bmm(hs.transpose(1, 2)) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        self.Wa = self.Wa.masked_fill(mask.unsqueeze(1), -10000)
        self.Wa = F.softmax(self.Wa, 2)
        self.V = self.Wa.bmm(hs) # [B, 1, L] @ [B, L, H] = [B, 1, H]
        return self.V

class copy(nn.Module): # copying mechanism
    def __init__(self):
        super().__init__()

        # architecture
        self.Wc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.V = None

    def forward(self, ht, hs, mask):
        hs = hs[:, :-1] # remove EOS token [B, L - 1, H]
        self.V = ht.bmm(self.Wc(hs).tanh().transpose(1, 2)) # [B, 1, L - 1]
        self.V = self.V.squeeze(1).masked_fill(mask[:, :-1], -10000)
        return self.V
