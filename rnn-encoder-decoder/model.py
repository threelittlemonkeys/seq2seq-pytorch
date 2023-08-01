from utils import *
from embedding import embed
import random

class rnn_encoder_decoder(nn.Module):

    def __init__(self, x_cti, x_wti, y_wti):

        super().__init__()

        # architecture
        self.enc = encoder(x_cti, x_wti)
        self.dec = decoder(x_wti, y_wti)
        if CUDA: self = self.cuda()

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        loss = Tensor(len(y0))
        mask, lens = maskset(xw)

        self.dec.M, self.dec.H = self.enc(xc, xw, lens)
        self.dec.h = zeros(len(y0), 1, HIDDEN_SIZE)
        yi = LongTensor([SOS_IDX] * len(y0))

        for t in range(y0.size(1)):
            yo = self.dec(xw, yi.unsqueeze(1), mask)
            yi = y0[:, t] # teacher forcing
            loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX, reduction = "none")

        loss /= y0.size(1) # average over timesteps

        return loss.mean(), loss

class encoder(nn.Module):

    def __init__(self, cti, wti):

        super().__init__()

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

    def init_state(self, b): # initialize states

        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "GRU":
            return hs
        cs = zeros(n, b, h) # LSTM cell state
        return (hs, cs)

    def forward(self, xc, xw, lens):

        s = self.init_state(len(xw))
        x = self.embed(xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first = True)
        h, s = self.rnn(x, s)
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
        if ATTN:
            self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
            self.Wo = nn.Linear(HIDDEN_SIZE, len(y_wti))
            self.softmax = nn.LogSoftmax(1)
        if COPY:
            self.Wo = nn.Linear(HIDDEN_SIZE, len(y_wti))
            self.copy = copy(x_wti, y_wti)

    def forward(self, xw, y1, mask):

        x = self.embed(None, y1)

        if ATTN:
            x = torch.cat((x, self.h), 2) # input feeding
            h, self.H = self.rnn(x, self.H)
            self.attn(self.M, h, mask)
            self.h = self.Wc(torch.cat((self.attn.V, h), 2)).tanh()
            h = self.Wo(self.h).squeeze(1) # [B, V]
            y = self.softmax(h)

        if COPY:
            self.attn(self.M, self.h, mask) # attentive read
            # selective read
            x = torch.cat((x, self.attn.V), 2)
            self.h, self.H = self.rnn(x, self.H)
            g = self.Wo(self.h).squeeze(1) # generation scores [B, V]
            c = self.copy.score(self.M, self.h, mask) # copy scores [B, L']
            y = self.copy.mix(xw, g, c) # [B, V']

        return y

class attn(nn.Module): # attention mechanism (Luong et al 2015)

    def __init__(self):

        super().__init__()

        # architecture
        self.W = None # attention weights
        self.V = None # context vector

    def forward(self, hs, ht, mask):

        a = ht.bmm(hs.transpose(1, 2)) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        a = a.masked_fill(mask.unsqueeze(1), -10000)
        self.W = F.softmax(a, 2) # [B, 1, L]
        self.V = self.W.bmm(hs) # [B, 1, L] @ [B, L, H] = [B, 1, H]

class copy(nn.Module): # copying mechanism (Gu et al 2016)

    def __init__(self, x_wti, y_wti):

        super().__init__()
        self.xyi = {i: y_wti[w] for w, i in x_wti.items() if w in y_wti}

        # architecture
        self.W = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def score(self, hs, ht, mask): # copy score function

        c = self.W(hs[:, :-1]).tanh() # [B, L' = L - 1, H]
        c = ht.bmm(c.transpose(1, 2)) # [B, 1, H] @ [B, H, L'] = [B, 1, L']
        c = c.squeeze(1).masked_fill(mask[:, :-1], -10000) # [B, L']

        return c

    def map(self, xw, vocab_size): # source to target index mapping

        idx = []
        oov = {}

        for i in xw.tolist():
            idx.append([])
            for j in i:
                if j in self.xyi:
                    j = self.xyi[j]
                else:
                    if j not in oov:
                        oov[j] = vocab_size + len(oov)
                    j = oov[j]
                idx[-1].append(j)

        idx = LongTensor(idx) # [B, L']
        m = zeros(*xw.size(), vocab_size + len(oov)) # [B, L', V' = V + OOV]
        m = m.scatter(2, idx.unsqueeze(2), 1) # one hot vector

        return m, len(oov)

    def mix(self, xw, g, c):

        xw = xw[:, :-1] # [B, L']
        m, oov_size = self.map(xw, g.size(1)) # [B, L', V']

        z = F.softmax(torch.cat([g, c], 1), 1) # normalization
        g, c = z.split([g.size(1), c.size(1)], 1)

        g = torch.cat([g, Tensor(len(g), oov_size).fill_(1e-6)], 1) # [B, V']
        c = c.unsqueeze(1).bmm(m) # [B, 1, L'] @ [B, L', V'] = [B, 1, V']
        z = (g + c.squeeze(1)).log() # mixed probabilities [B, V']

        return z
