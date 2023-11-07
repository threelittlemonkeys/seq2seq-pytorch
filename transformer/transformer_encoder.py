from utils import *
from embedding import *
from transformer_modules import *

class transformer_encoder(nn.Module):

    def __init__(self, x_cti, x_wti):

        super().__init__()

        # architecture
        self.embed = nn.Embedding(len(x_wti), EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pos_encoding() # positional encoding
        self.dropout = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([transformer_encoder_layer() for _ in range(NUM_LAYERS)])

    def forward(self, xc, xw, x_mask):

        x = self.embed(xw)
        h = x + self.pe(x.size(1))
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, x_mask)

        return h

class transformer_encoder_layer(nn.Module):

    def __init__(self):

        super().__init__()

        # architecture
        self.attn = mh_attn() # multi-head self-attention
        self.ffn = ffn(2048) # position-wise feed-forward network

    def forward(self, x, x_mask):

        h = self.attn(x, x, x, x_mask)
        h = self.ffn(h)

        return h
