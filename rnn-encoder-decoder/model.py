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
        self.dec.M, self.dec.H = self.enc(b, xc, xw, lens)
        self.dec.h = zeros(b, 1, HIDDEN_SIZE)
        if ATTN:
            self.dec.attn.V = zeros(b, 1, HIDDEN_SIZE)
        if COPY:
            self.dec.attn.V = zeros(b, 1, HIDDEN_SIZE)
            self.dec.copy.V = zeros(b, 1, HIDDEN_SIZE)
        yi = LongTensor([SOS_IDX] * b)

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

    def init_state(self, b): # initialize RNN states

        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, b, xc, xw, lens):

        s = self.init_state(b)
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
        if ATTN:
            self.attn = attn()
            self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        if COPY:
            self.attn = attn()
            self.copy = copy(x_wti, y_wti)
        self.Wo = nn.Linear(HIDDEN_SIZE, len(y_wti))
        self.softmax = nn.LogSoftmax(1)

    def forward(self, xw, y1, mask):

        x = self.embed(None, y1)

        if ATTN:
            x = torch.cat((x, self.h), 2) # input feeding
            h, self.H = self.rnn(x, self.H)
            self.attn(self.M, h, mask)
            self.h = self.Wc(torch.cat((self.attn.V, h), 2)).tanh()
            h = self.Wo(self.h).squeeze(1)
            y = self.softmax(h)
            return y

        if COPY:
            self.attn(self.M, self.h, mask) # attentive read
            # selective read
            x = torch.cat((x, self.attn.V), 2)
            self.h, self.H = self.rnn(x, self.H)
            g = self.Wo(self.h).squeeze(1) # generation scores
            c = self.copy(self.M, self.h, mask) # copy scores
            h = self.copy.merge(xw, g, c)
            y = self.softmax(h)
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
        self.W = F.softmax(a, 2)
        self.V = self.W.bmm(hs) # [B, 1, L] @ [B, L, H] = [B, 1, H]

class copy(nn.Module): # copying mechanism (Gu et al 2016)

    def __init__(self, x_wti, y_wti):

        super().__init__()
        self.xyi = {i: y_wti[w] for w, i in x_wti.items() if w in y_wti}

        # architecture
        self.W = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, hs, ht, mask): # copy scores

        hs = hs[:, :-1] # remove EOS token [B, L' = L - 1, H]
        c = ht.bmm(self.W(hs).tanh().transpose(1, 2)) # [B, 1, H] @ [B, H, L'] = [B, 1, L']
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
        m = zeros(*xw.size(), vocab_size + len(oov)).detach() # [B, L', V + OOV]
        m = m.scatter(2, idx.unsqueeze(2), 1)

        return m, len(oov)

    def merge(self, xw, g, c):

        xw = xw[:, :-1] # [B, L']
        m, oov_size = self.map(xw, g.size(1)) # [B, L', V + OOV]

        z = F.softmax(torch.cat([g, c], 1), 1) # combined scores
        g, c = z.split([g.size(1), c.size(1)], 1)

        g = torch.cat([g, zeros(g.size(0), oov_size)], 1) # [B, V + OOV]
        c = c.unsqueeze(1).bmm(m) # [B, 1, L'] @ [B, L', V + OOV] = [B, 1, V + OOV]
        z = g + c.squeeze() # [B, V + OOV]

        return z
