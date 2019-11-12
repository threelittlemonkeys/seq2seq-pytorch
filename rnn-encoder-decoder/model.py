from utils import *
from embedding import embed

class rnn_enc_dec(nn.Module):
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
        mask, lens = maskset(y0 if HRE else xw)
        self.dec.enc_out = self.enc(b, xc, xw, lens)
        self.dec.hidden = self.enc.hidden
        dec_in = LongTensor([SOS_IDX] * b)
        if self.dec.feed_input:
            self.dec.attn.h = zeros(b, 1, HIDDEN_SIZE)
        for t in range(y.size(1)):
            dec_out = self.dec(dec_in.unsqueeze(1), enc_out, t, mask)
            dec_in = y[:, t] # teacher forcing
            loss += F.nll_loss(dec_out, dec_in, ignore_index = PAD_IDX)
        loss /= y.size(1) # divide by senquence length
        # loss /= y.gt(0).sum().float() # divide by the number of unpadded tokens
        return loss

    def decode(self, x): # for inference
        pass

class encoder(nn.Module):
    def __init__(self, cti_size, wti_size):
        super().__init__()

        # architecture
        self.embed = embed(cti_size, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = sum(EMBED.values()),
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
        self.hidden = self.init_state(b)
        x = self.embed(xc, xw)
        if HRE: # [B * doc_len, 1, H] -> [B, doc_len, H]
            x = x.view(b, -1, EMBED_SIZE)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h

class decoder(nn.Module):
    def __init__(self, wti_size):
        super().__init__()
        self.hidden = None # hidden state
        self.enc_out = None # encoder output
        self.dec_out = None # decoder output
        self.feed_input = True # input feeding

        # architecture
        self.embed = embed(-1, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = sum(EMBED.values()) + (HIDDEN_SIZE if self.feed_input else 0),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        self.out = nn.Linear(HIDDEN_SIZE, wti_size)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, dec_in, enc_out, t, mask):
        x = self.embed(None, dec_in)
        if self.feed_input:
            x = torch.cat((x, self.attn.h), 2)
        h, _ = self.rnn(x, self.hidden)
        if self.attn:
            h = self.attn(h, enc_out, t, mask)
        h = self.out(h).squeeze(1)
        y = self.softmax(h)
        return y

class attn(nn.Module): # attention layer (Luong et al 2015)
    def __init__(self):
        super().__init__()
        self.type = "global" # global, local-m, local-p
        self.method = "dot" # dot, general, concat
        self.h = None # attention vector (for input feeding)
        self.w = None # attention weights (for visualization)

        # architecture
        if self.type in ("local-m", "local-p"):
            self.window_size = 5
            if self.type[-1] == "p":
                self.Wp = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
                self.Vp = nn.Linear(HIDDEN_SIZE, 1)
        if self.method == "general":
            self.Wa = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        elif self.method  == "concat":
            pass # TODO
        self.softmax = nn.Softmax(2)
        self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)

    def window(self, ht, hs, t, mask): # TODO for local attention
        if self.type[-1] == "m": # monotonic
            p0 = max(0, min(t - self.window_size, hs.size(1) - self.window_size))
            p1 = min(hs.size(1), t + 1 + self.window_size)
            return hs[:, p0:p1], mask[0][:, p0:p1], None
        if self.type[-1] == "p": # predicative
            S = Tensor(mask[1]) # source sequence length
            pt = S * torch.sigmoid(self.Vp(torch.tanh(self.Wp(ht)))).view(-1) # aligned position
            hs_w = []
            mask_w = []
            k = [] # truncated Gaussian distribution as kernel function
            for i in range(BATCH_SIZE):
                p = int(S[i].item())
                seq_len = mask[1][i]
                min_len = mask[1][-1]
                p0 = max(0, min(p - self.window_size, seq_len - self.window_size))
                p1 = min(seq_len, p + 1 + self.window_size)
                if min_len < p1 - p0:
                    p0 = 0
                    p1 = min_len
                hs_w.append(hs[i, p0:p1])
                mask_w.append(mask[0][i, p0:p1])
                sd = (p1 - p0) / 2 # standard deviation
                v = [torch.exp(-(j - pt[i]).pow(2) / (2 * sd ** 2)) for j in range(p0, p1)]
                k.append(torch.cat(v))
            hs_w = torch.cat(hs_w).view(BATCH_SIZE, -1, HIDDEN_SIZE)
            mask_w = torch.cat(mask_w).view(BATCH_SIZE, -1)
            k = torch.cat(k).view(BATCH_SIZE, 1, -1)
            return hs_w, mask_w, k

    def align(self, ht, hs, mask, k):
        if self.method == "dot":
            a = ht.bmm(hs.transpose(1, 2))
        elif self.method == "general":
            a = ht.bmm(self.Wa(hs).transpose(1, 2))
        elif self.method == "concat":
            pass # TODO
        a = a.masked_fill(mask.unsqueeze(1), -10000) # masking in log space
        a = self.softmax(a) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        if self.type == "local-p":
            a = a * k
        return a # alignment vector as attention weights

    def forward(self, ht, hs, t, mask):
        if self.type in ("local-m", "local-p"):
            hs, mask, k = self.window(ht, hs, t, mask)
        else:
            k = None
            mask = mask[0]
        a = self.align(ht, hs, mask, k)
        c = a.bmm(hs) # context vector [B, 1, L] @ [B, L, H] = [B, 1, H]
        h = torch.tanh(self.Wc(torch.cat((c, ht), 2)))
        self.a = a
        self.h = h
        return h # attention vector as attentional hidden state
