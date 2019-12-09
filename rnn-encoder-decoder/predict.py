from model import *
from utils import *

def load_model():
    x_cti = load_tkn_to_idx(sys.argv[2])
    x_wti = load_tkn_to_idx(sys.argv[3])
    y_itw = load_idx_to_tkn(sys.argv[4])
    model = rnn_enc_dec(len(x_cti), len(x_wti), len(y_itw))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, x_cti, x_wti, y_itw

def greedy_search(dec, yo, batch, itw, eos, lens):
    p, yi = yo.topk(1)
    y = yi.view(-1).tolist()
    for i, _ in filter(lambda x: not x[1], enumerate(eos)):
        eos[i] = (y[i] == EOS_IDX)
        batch.y1[i].append(itw[y[i]])
        batch.prob[i] += p[i]
        batch.attn[i].append([itw[y[i]], *dec.attn.w[i][0].tolist()])
    return yi

def beam_search(dec, yo, batch, itw, eos, lens, t):
    bp, by = dec_out[:len(eos)].topk(BEAM_SIZE) # [B * BEAM_SIZE, BEAM_SIZE]
    bp += Tensor([-10000 if b else a[4] for a, b in zip(batch, eos)]).unsqueeze(1) # update
    bp = bp.view(-1, BEAM_SIZE ** 2) # [B, BEAM_SIZE * BEAM_SIZE]
    by = by.view(-1, BEAM_SIZE ** 2)
    if t == 0: # remove non-first duplicate beams
        bp = bp[:, :BEAM_SIZE]
        by = by[:, :BEAM_SIZE]
    for i, (p, y) in enumerate(zip(bp, by)): # for each sequence
        j = i * BEAM_SIZE
        b1, m1 = [], [] # batch and heatmap to be updated
        if VERBOSE >= 2:
            for k in range(0, len(p), BEAM_SIZE): # for each beam
                q = j + k // BEAM_SIZE
                w = [(next(reversed(batch[q][3]), SOS_IDX), batch[q][4])] # previous token
                w += list(zip(y, p))[k:k + BEAM_SIZE] # current candidates
                w = [(itw[a], round(b.item(), 4)) for a, b in w]
                print("beam[%d][%d][%d] =" % (i, t, q), w[0], "->", w[1:])
        for p, k in zip(*p.topk(BEAM_SIZE)): # for each n-best candidate
            q = j + k // BEAM_SIZE
            b1.append(batch[q].copy())
            b1[-1][3] = b1[-1][3] + [y[k]] # word
            b1[-1][4] = p # probability
            m1.append(heatmap[q].copy())
            m1[-1].append([itw[y[k]]] + dec.attn.w[q][0][:len(batch[j][1]) + 1].tolist())
        for k in filter(lambda x: eos[j + x], range(BEAM_SIZE)): # append completed sequences
            b1.append(batch[j + k])
            m1.append(heatmap[j + k])
        topk = sorted(zip(b1, m1), key = lambda x: -x[0][4])[:BEAM_SIZE]
        for k, (b1, m1) in enumerate(topk, j):
            batch[k] = b1
            eos[k] = (b1[3][-1] == EOS_IDX)
            heatmap[k] = m1
            if VERBOSE >= 2:
                print("output[%d][%d][%d] = " % (i, t, k), end = "")
                print(([itw[x] for x in b1[3]], round(b1[4].item(), 4)))
        if VERBOSE >= 2:
            print()
    return LongTensor([next(reversed(x[3]), SOS_IDX) for x in batch]).unsqueeze(1)

def run_model(model, data, itw):
    data.sort()
    for batch in data.split():
        b, t = len(batch.x0), 0 # batch size, time step
        xc, xw = data.tensor(batch.xc, batch.xw, batch.lens, eos = True)
        eos = [False for _ in batch.x0] # EOS states
        mask, lens = maskset([x + 1 for x in batch.lens] if HRE else xw)
        model.dec.hs = model.enc(b, xc, xw, lens)
        model.dec.hidden = model.enc.hidden
        yi = LongTensor([[SOS_IDX]] * b)
        if model.dec.feed_input:
            model.dec.attn.v = zeros(b, 1, HIDDEN_SIZE)
        for i in range(len(batch.lens)): # attention heatmap column headers
            batch.attn[i].append(["", *batch.x1[i], EOS])
        while sum(eos) < len(eos) and t < MAX_LEN:
            yo = model.dec(yi, mask, t)
            args = (model.dec, yo, batch, itw, eos, lens)
            yi = greedy_search(*args) if BEAM_SIZE == 1 else beam_search(*args, t)
        data.y1.extend(batch.y1)
        data.prob.extend(batch.prob)
        data.attn.extend(batch.attn)
    data.unsort()
    if VERBOSE:
        print()
        for i, x in filter(lambda x: not x[0] % BEAM_SIZE, enumerate(data.attn)):
            print("attn[%d] =" % (i // BEAM_SIZE))
            print(mat2csv(x, rh = True))
    for i, (x0, y0, y1) in enumerate(zip(data.x0, data.y0, data.y1)):
        if not i % BEAM_SIZE: # use the best candidate from each beam
            y1.pop() # remove EOS token
            yield x0, y0, y1

def predict(filename, model, x_cti, x_wti, y_itw):
    data = dataloader()
    fo = open(filename)
    for x0 in fo:
        x0 = x0.strip()
        x1 = tokenize(x0, UNIT)
        xc = [[x_cti[c] if c in x_cti else UNK_IDX for c in w] for w in x1]
        xw = [x_wti[w] if w in x_wti else UNK_IDX for w in x1]
        data.append_item(x0 = [x0], x1 = [x1], xc = [xc], xw = [xw])
        for _ in range(BEAM_SIZE - 1):
            data.append_row()
            data.append_item(x0 = [x0], x1 = [x1], xc = [xc], xw = [xw])
        data.append_row()
    fo.close()
    data.strip()
    with torch.no_grad():
        model.eval()
        return run_model(model, data, y_itw)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src.char_to_idx vocab.src.word_to_idx vocab.tgt.word_to_idx test_data" % sys.argv[0])
    for x, y0, y1 in predict(sys.argv[5], *load_model()):
        print((x, y0, y1) if y0 else (x, y1))
