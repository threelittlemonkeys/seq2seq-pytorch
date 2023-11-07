from utils import *
from transformer_encoder import *
from transformer_decoder import *

class transformer(nn.Module):

    def __init__(self, x_cti, x_wti, y_wti):

        super().__init__()

        # architecture
        self.enc = transformer_encoder(x_cti, x_wti)
        self.dec = transformer_decoder(y_wti)
        if CUDA: self = self.cuda()

    def init_state(self, b):

        for i in range(NUM_LAYERS):
            self.dec.layers[i].attn2.W = zeros(b, NUM_HEADS, self.dec.M.size(1))

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        b = len(xw) # batch size
        yi = y0[:, :-1]

        x_mask = padding_mask(xw, xw) # [B, 1, Lx, Lx]
        y_mask = padding_mask(yi, yi) | lookahead_mask(yi, yi) # [B, 1, Ly, Ly]
        xy_mask = padding_mask(yi, xw) # [B, 1, Ly, Lx]

        self.dec.M = self.enc(xc, xw, x_mask)
        self.init_state(b)
        y1 = self.dec(yi, y_mask, xy_mask).flatten(0, 1) # [B * L, V]
        y0 = y0[:, 1:].reshape(-1)
        loss = F.nll_loss(y1, y0, ignore_index = PAD_IDX)

        return loss
