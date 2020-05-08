from model import *
from utils import *
from dataloader import *
from beamsearch import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2])
    wti = load_tkn_to_idx(sys.argv[3])
    model = ptrnet(len(cti), len(wti))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti

'''
def beam_search(dec, y1, batch, eos, lens, t):
    bp, by = y1.topk(BEAM_SIZE) # [B * BEAM_SIZE, BEAM_SIZE]
    bp += Tensor([-10000 if b else a for a, b in zip(batch.prob, eos)]).unsqueeze(1)
    bp = bp.view(-1, BEAM_SIZE ** 2) # [B, BEAM_SIZE * BEAM_SIZE]
    by = by.view(-1, BEAM_SIZE ** 2)
    if t == 0: # remove non-first duplicate beams
        bp = bp[:, :BEAM_SIZE]
        by = by[:, :BEAM_SIZE]
    for i, (p, y) in enumerate(zip(bp, by)): # for each sequence
        j, y = i * BEAM_SIZE, y.tolist()
        _y, _p, _a = [], [], []
        if VERBOSE >= 2:
            print()
            for k in range(0, len(p), BEAM_SIZE): # for each beam
                q = j + k // BEAM_SIZE
                w = [(batch.y1[q], batch.prob[q])] # previous token
                w += list(zip(y, p))[k:k + BEAM_SIZE] # current candidates
                w = [(a, round(b.item(), 4)) for a, b in w]
                print("beam[%d][%d][%d] =" % (i, t, k // BEAM_SIZE), w[0], "->", *w[1:])
        for p, k in zip(*p.topk(BEAM_SIZE)): # n-best candidates
            q = j + k // BEAM_SIZE
            _y.append(batch.y1[q] + [y[k]])
            _p.append(batch.prob[q] + p)
            _a.append(batch.attn[q] + [[y[k], *dec.attn.w[q][:lens[q]].exp()]])
        for k in filter(lambda x: eos[j + x], range(BEAM_SIZE)): # completed sequences
            _y.append(batch.y1[j + k])
            _p.append(batch.prob[j + k])
            _a.append(batch.attn[j + k])
        topk = sorted(zip(_y, _p, _a), key = lambda x: -x[1])[:BEAM_SIZE]
        for k, (y, p, a) in enumerate(topk, j):
            eos[k] = (y[-1] == lens[j] - 1 or y[-1] in y[:-1])
            batch.y1[k], batch.prob[k], batch.attn[k] = y, p, a
            if VERBOSE >= 2:
                print("best[%d] =" % (k - j), (y, round(p.item(), 4)))
    return LongTensor([next(reversed(x), SOS_IDX) for x in batch.y1]).unsqueeze(1)
'''

def run_model(model, data):
    with torch.no_grad():
        model.eval()
        for batch in data.split():
            xc, xw, lens = batch.sort()
            xc, xw = data.tensor(xc, xw, lens, eos = True)
            b, t = len(xw), 0 # batch size, time step
            eos = [False for _ in xw] # EOS states
            mask, lens = maskset(xw)
            model.dec.hs = model.enc(b, xc, xw, lens)
            model.dec.hidden = model.enc.hidden
            yc = LongTensor([[[SOS_IDX]]] * b)
            yw = LongTensor([[SOS_IDX]] * b)
            batch.y1 = [[] for _ in range(b)]
            batch.prob = [Tensor([0]) for _ in range(b)]
            batch.attn = [[["", *batch.x1[i], EOS]] for i in batch.idx]
            while t < MAX_LEN and sum(eos) < len(eos):
                yo = model.dec(yc, yw, mask)
                args = (model.dec, batch, eos, lens, yo)
                yw = beam_search(*args, t) if BEAM_SIZE > 1 else greedy_search(*args)
                yc = torch.cat([xc[i, j] for i, j in enumerate(yw)]).unsqueeze(1)
                t += 1
            batch.unsort()
            if VERBOSE:
                print()
                for i, x in filter(lambda x: not x[0] % BEAM_SIZE, enumerate(batch.attn)):
                    print("attn[%d] =" % (i // BEAM_SIZE))
                    print(mat2csv(x, rh = True))
            for i, (x0, y0, y1) in enumerate(zip(batch.x0, batch.y0, batch.y1)):
                if not i % BEAM_SIZE: # use the best candidate from each beam
                    y1.pop() # remove EOS token
                    yield x0, y0, y1

def predict(filename, model, cti, wti):
    data = dataloader()
    with open(filename) as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        for x0 in block.split("\n"):
            if re.match("[^\t]+\t[0-9]+( [0-9]+)*$", x0):
                x0, y0 = x0.split("\t")
                y0 = [int(x) for x in y0.split(" ")]
            else: # no ground truth provided
                y0 = []
            x1 = tokenize(x0)
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            data.append_item(x0, x1, xc, xw, y0)
        for _ in range(BEAM_SIZE - 1):
            data.x0.append(data.x0[-1])
            data.x1.append(data.x1[-1])
            data.xc.append(data.xc[-1])
            data.xw.append(data.xw[-1])
            data.y0.append(data.y0[-1])
        data.append_row()
    data.strip()
    return run_model(model, data)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model char_to_idx word_to_idx test_data" % sys.argv[0])
    for x, y0, y1 in predict(sys.argv[4], *load_model()):
        print((x, y0, y1) if y0 else (x, y1))
