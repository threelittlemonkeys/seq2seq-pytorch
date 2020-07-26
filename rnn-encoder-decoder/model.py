from utils import *
from embedding import embed

class rnn_encoder_decoder(nn.Module):
    def __init__(self, x_cti, x_wti, y_wti):
        super().__init__()

        # architecture
        self.enc = encoder(x_cti, x_wti)
        self.dec = decoder(x_wti, y_wti)
        self = self.cuda() if CUDA else self

    def forward(self, xc, xw, y0): # for training
        b = y0.size(0) # batch size
        loss = Tensor(b)
        self.zero_grad()
        mask, lens = maskset(xw)
        self.dec.M, self.dec.h = self.enc(b, xc, xw, lens)
        self.dec.H = self.enc.H
        self.dec.attn.V = zeros(b, 1, HIDDEN_SIZE)
        if COPY: self.dec.copy.V = zeros(b, 1, HIDDEN_SIZE)
        yi = LongTensor([SOS_IDX] * b)
        for t in range(y0.size(1)):
            yo = self.dec(xw, yi.unsqueeze(1), mask)
            yi = y0[:, t] # teacher forcing
            loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX, reduction = "none")
        loss /= y0.size(1) # average over timesteps
        return loss.mean(), loss

    def decode(self, x): # for inference
        pass

class encoder(nn.Module):
    def __init__(self, cti, wti):
        super().__init__()
        self.H = None # encoder hidden states

        # architecture
        self.embed = embed(ENC_EMBED, len(cti), len(wti))
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
    def __init__(self, x_wti, y_wti):
        super().__init__()
        self.M = None # encoder hidden states
        self.H = None # decoder hidden states
        self.h = None # decoder output

        # architecture
        self.embed = embed(DEC_EMBED, 0, len(y_wti))
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim + HIDDEN_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        if COPY: self.copy = copy(x_wti, y_wti)
        self.Wo = nn.Linear(HIDDEN_SIZE, len(y_wti))
        self.softmax = nn.LogSoftmax(1)

    def forward(self, xw, y1, mask):
        x = self.embed(None, y1)

        if ATTN:
            x = torch.cat((x, self.attn.V), 2) # input feeding
            h, _ = self.rnn(x, self.H)
            self.attn.V = self.attn(self.M, h, mask)
            h = self.Wc(torch.cat((self.attn.V, h), 2)).tanh()
            h = self.Wo(h).squeeze(1)
            y = self.softmax(h)
            return y

        if COPY:
            self.attn.V = self.attn(self.M, self.h, mask)
            x = torch.cat((x, self.attn.V), 2)
            self.h, _ = self.rnn(x, self.H)
            g = self.Wo(self.h).squeeze(1) # generation scores
            c = self.copy(self.M, self.h, mask) # copy scores
            h = self.copy.merge(xw, g, c)
            y = self.softmax(h)
            return y

class attn(nn.Module): # attention mechanism
    def __init__(self):
        super().__init__()

        # architecture
        self.Wa = None # attention weights
        self.V = None # context vector

    def forward(self, hs, ht, mask):
        a = ht.bmm(hs.transpose(1, 2)) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        a = a.masked_fill(mask.unsqueeze(1), -10000)
        self.Wa = F.softmax(a, 2)
        return self.Wa.bmm(hs) # [B, 1, L] @ [B, L, H] = [B, 1, H]

class copy(nn.Module): # copying mechanism
    def __init__(self, x_wti, y_wti):
        super().__init__()
        self.stt = {i: y_wti[w] for w, i in x_wti.items() if w in y_wti} # source to target
        self.vocab_size = len(y_wti) # target vocaublary size (V)

        # architecture
        self.Wc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.V = None

    def forward(self, hs, ht, mask):
        hs = hs[:, :-1] # remove EOS token [B, L - 1, H]
        self.V = ht.bmm(self.Wc(hs).tanh().transpose(1, 2)) # [B, 1, L - 1]
        self.V = self.V.squeeze(1).masked_fill(mask[:, :-1], -10000)
        return self.V

    def map(self, args): # source sequence mapping [L] -> [V + L]
        i, x = args
        if x > UNK_IDX and x in self.stt:
            return self.stt[x]
        return self.vocab_size + i

    def merge(self, xw, g, c):
        _b, _g, _c = len(xw), g.size(1), c.size(1)
        # h = F.softmax(torch.cat([g, c], 1), 1)
        # g, c = h.split([_g, _c], 1)
        idx = [list(map(self.map, enumerate(x[:-1]))) for x in xw.tolist()]
        g = torch.cat([g, zeros(c.size())], 1)
        c = zeros(_b, _g + _c).scatter(1, LongTensor(idx), c)
        return g + c # [B, V + L]
