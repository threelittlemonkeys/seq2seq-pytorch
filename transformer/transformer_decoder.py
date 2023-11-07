from utils import *
from embedding import *
from transformer_modules import *

class transformer_decoder(nn.Module):

    def __init__(self, y_wti):

        super().__init__()
        self.M = None # encoder hidden states

        # architecture
        self.embed = nn.Embedding(len(y_wti), EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pos_encoding() # positional encoding
        self.dropout = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([transformer_decoder_layer() for _ in range(NUM_LAYERS)])
        self.out = nn.Linear(EMBED_SIZE, len(y_wti))
        self.softmax = nn.LogSoftmax(2)

    def forward(self, yi, y_mask, xy_mask):

        x = self.embed(yi)
        h = x + self.pe(x.size(1))
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(self.M, h, y_mask, xy_mask)

        h = self.out(h)
        yo = self.softmax(h)

        return yo

class transformer_decoder_layer(nn.Module):

    def __init__(self):

        super().__init__()

        # architecture
        self.attn1 = mh_attn() # masked multi-head self-attention
        self.attn2 = mh_attn() # multi-head cross-attention
        self.ffn = ffn(2048)

    def forward(self, x, y, y_mask, xy_mask):

        h = self.attn1(y, y, y, y_mask)
        h = self.attn2(h, x, x, xy_mask)
        h = self.ffn(h)

        return h
