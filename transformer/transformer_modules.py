from utils import *

def padding_mask(q, k): # mask out padded positions

    q = q.eq(PAD_IDX).unsqueeze(2).repeat(1, 1, k.size(1))
    k = k.eq(PAD_IDX).unsqueeze(1).repeat(1, q.size(1), 1)

    return (q | k).unsqueeze(1)

def lookahead_mask(q, k): # mask out subsequent positions

    return triu(torch.ones(q.size(1), k.size(1)), 1).to(dtype = torch.bool)

class pos_encoding(nn.Module): # positional encoding

    def __init__(self, maxlen = 1000):

        super().__init__()

        self.pe = Tensor(maxlen, EMBED_SIZE)
        pos = torch.arange(0, maxlen, 1.).unsqueeze(1)
        k = torch.exp(np.log(10000) * -torch.arange(0, EMBED_SIZE, 2.) / EMBED_SIZE)
        self.pe[:, 0::2] = torch.sin(pos * k)
        self.pe[:, 1::2] = torch.cos(pos * k)

    def forward(self, n):

        return self.pe[:n]

class mh_attn(nn.Module): # multi-head attention

    def __init__(self):

        super().__init__()
        self.W = None # attention weights

        # architecture
        self.norm = nn.LayerNorm(EMBED_SIZE) # pre-layer normalization
        self.Wq = nn.Linear(EMBED_SIZE, NUM_HEADS * DK) # query
        self.Wk = nn.Linear(EMBED_SIZE, NUM_HEADS * DK) # key for attention distribution
        self.Wv = nn.Linear(EMBED_SIZE, NUM_HEADS * DV) # value for context representation
        self.Wo = nn.Linear(NUM_HEADS * DV, EMBED_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def sdp_attn(self, q, k, v, mask): # scaled dot-product attention

        a = q.matmul(k.transpose(2, 3)) / np.sqrt(DK) # compatibility function
        a = a.masked_fill(mask, -10000)
        a = self.W = F.softmax(a, 3) # attention weights [B, NUM_HEADS, L, L]
        a = a.matmul(v) # [B, NUM_HEADS, L, DV]

        return a

    def forward(self, q, k, v, mask):

        b = q.size(0)
        x = q # identity
        q = self.norm(q) # [B, L, H]
        k = self.norm(k)
        v = self.norm(v)
        q = self.Wq(q).view(b, -1, NUM_HEADS, DK).transpose(1, 2) # [B, NUM_HEADS, L, D]
        k = self.Wk(k).view(b, -1, NUM_HEADS, DK).transpose(1, 2)
        v = self.Wv(v).view(b, -1, NUM_HEADS, DV).transpose(1, 2)
        h = self.sdp_attn(q, k, v, mask) # [B, NUM_HEADS, L, DV]
        h = h.transpose(1, 2).flatten(2, 3) # [B, L, H]
        h = self.Wo(h)
        h = self.dropout(h)
        h = x + h # residual connection [B, L, H]

        return h

class ffn(nn.Module): # position-wise feed-forward networks

    def __init__(self, dim):

        super().__init__()

        # architecture
        self.norm = nn.LayerNorm(EMBED_SIZE) # pre-layer normalization
        self.layers = nn.Sequential(
            nn.Linear(EMBED_SIZE, dim),
            nn.ReLU(),
            nn.Linear(dim, EMBED_SIZE),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        h = self.norm(x)
        h = self.layers(h)
        h = self.dropout(h)
        h = x + h # residual connection [B, L, H]

        return h
