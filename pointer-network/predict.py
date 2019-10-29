from model import *
from utils import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2])
    wti = load_tkn_to_idx(sys.argv[3])
    model = ptrnet(len(cti), len(wti))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti

def greedy_search(dec, data, eos, mask):
    bp, by = dec.dec_out.topk(1)
    y = by.view(-1).tolist()
    for i, _ in filter(lambda x: not x[1], enumerate(eos)):
        j = mask[1][i] # sequence length
        eos[i] = (y[i] == j - 1 or y[i] in data._y1[i])
        data._y1[i].append(y[i])
        data._prob[i] += bp[i]
        data._attn[i].append([y[i], *dec.attn.a[i, :j].exp()])
    return by

def beam_search(dec, data, eos, mask, t):
    bp, by = dec.dec_out.topk(BEAM_SIZE) # [B * BEAM_SIZE, BEAM_SIZE]
    bp += Tensor([-10000 if b else a for a, b in zip(data._prob, eos)]).unsqueeze(1) # update
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
                w = [(next(reversed(data._y1[q]), SOS_IDX), data._prob[q])] # previous token
                w += list(zip(y, p))[k:k + BEAM_SIZE] # current candidates
                w = [(a, round(b.item(), 4)) for a, b in w]
                print("batch[%d][%d][%d] =" % (i, t, k // BEAM_SIZE), w[0], "->", *w[1:])
        for p, k in zip(*p.topk(BEAM_SIZE)): # n-best candidates
            q = j + k // BEAM_SIZE
            _y.append(data._y1[q] + [y[k]])
            _p.append(data._prob[q] + p)
            _a.append(data._attn[q] + [[y[k], *dec.attn.a[q][:mask[1][q]].exp()]])
        for k in filter(lambda x: eos[j + x], range(BEAM_SIZE)): # completed sequences
            _y.append(data._y1[j + k])
            _p.append(data._prob[j + k])
            _a.append(data._attn[j + k])
        topk = sorted(zip(_y, _p, _a), key = lambda x: -x[1])[:BEAM_SIZE]
        for k, (y, p, a) in enumerate(topk, j):
            eos[k] = (y[-1] == mask[1][j] - 1 or y[-1] in y[:-1])
            data._y1[k], data._prob[k], data._attn[k] = y, p, a
            if VERBOSE >= 2:
                print("candidate[%d] =" % (k - j), (y, round(p.item(), 4)))
    return LongTensor([next(reversed(x), SOS_IDX) for x in data._y1]).unsqueeze(1)

def run_model(model, data):
    data.sort()
    for _ in data.split():
        b, t = len(data._x0), 0 # batch size, time step
        eos = [False for _ in data._x0] # EOS states
        lens = [len(x) for x in self._xw] if HRE else None # TODO
        xc, xw = data.tensor(data._xc, data._xw, _eos = True, doc_lens = lens)
        mask = None if HRE else maskset(xw) # TODO
        model.dec.enc_out = model.enc(b, xc, xw, mask)
        model.dec.hidden = model.enc.hidden
        yc = LongTensor([[[SOS_IDX]]] * b)
        yw = LongTensor([[SOS_IDX]] * b)
        for i, x in enumerate(data._x1): # attention heatmap column headers
            data._attn[i].append(["", *x, EOS])
        while sum(eos) < len(eos) and t < MAX_LEN:
            model.dec.dec_out = model.dec(yc, yw, mask)
            args = (model.dec, data, eos, mask)
            yw = greedy_search(*args) if BEAM_SIZE == 1 else beam_search(*args, t)
            yc = torch.cat([xc[i, j] for i, j in enumerate(yw)]).unsqueeze(1)
            t += 1
        data.y1.extend(data._y1)
        data.prob.extend(data._prob)
        data.attn.extend(data._attn)
    data.unsort()
    if VERBOSE:
        print()
        for i, x in filter(lambda x: not x[0] % BEAM_SIZE, enumerate(data.attn)):
            print("heatmap[%d] =" % (i // BEAM_SIZE)) # TODO
            print(mat2csv(x, rh = True))
    for i, (x0, y0, y1) in enumerate(zip(data.x0, data.y0, data.y1)):
        if i % BEAM_SIZE:
            continue
        y1.pop() # remove EOS token
        if HRE:
            pass
        else:
            yield x0, y0, y1

def predict(filename, model, cti, wti):
    data = dataset()
    fo = open(filename)
    for line in fo:
        x0 = line.strip()
        if x0:
            if re.match("[^\t]+\t[0-9]+( [0-9]+)*$", x0):
                x0, y0 = x0.split("\t")
                y0 = [int(x) for x in y0.split(" ")]
            else: # no ground truth provided
                y0 = []
            x1 = tokenize(x0)
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)
        if not (HRE and x0): # delimiters (\n, \n\n)
            for _ in range(BEAM_SIZE - 1):
                data.append_row()
                data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)
            data.append_row()
    fo.close()
    data.strip()
    with torch.no_grad():
        model.eval()
        return run_model(model, data)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model char_to_idx word_to_idx test_data" % sys.argv[0])
    for x, y0, y1 in predict(sys.argv[4], *load_model()):
        print((x, y0, y1) if y0 else (x, y1))
