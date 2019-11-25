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

def greedy_search(dec, yo, batch, eos, lens):
    p, dec_in = dec_out.topk(1)
    y = dec_in.view(-1).tolist()
    for i in range(len(eos)):
        if eos[i]:
            continue
        batch[i][3].append(y[i])
        batch[i][4] += p[i]
        eos[i] = (y[i] == EOS_IDX)
        heatmap[i].append([itw[y[i]]] + dec.attn.a[i][0].tolist())
    return dec_in

def beam_search(dec, yo, batch, eos, lens, t):
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
            m1[-1].append([itw[y[k]]] + dec.attn.a[q][0][:len(batch[j][1]) + 1].tolist())
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

def run_model(model, data):
    data.sort()
    for batch in data.split():
        b, t = len(batch.x0), 0 # batch size, time step
        xc, xw = data.tensor(batch.xc, batch.xw, batch.lens, eos = True)
        eos = [False for _ in batch.x0] # EOS states
        mask, lens = maskset([x + 1 for x in batch.lens] if HRE else xw)
        model.dec.hs = model.enc(b, xc, xw, lens)
        model.dec.hidden = model.enc.hidden
        yi = LongTensor([SOS_IDX] * b)
        if model.dec.feed_input:
            model.dec.attn.v = zeros(b, 1, HIDDEN_SIZE)
        for i in range(len(batch.lens)): # attention heatmap column headers
            batch.attn[i].append(["", *batch.x1[i], EOS])
        while sum(eos) < len(eos) and t < MAX_LEN:
            yo = model.dec(yi.unsqueeze(1), t, mask)
            args = (model.dec, yo, batch, eos, lens)
            yi = greedy_search(*args) if BEAM_SIZE == 1 else beam_search(*args, t)
            break
    exit()
    t = 0
    eos = [False for _ in batch] # number of completed sequences in the batch
    while len(batch) < BATCH_SIZE:
        batch.append([-1, [], [EOS_IDX], [], 0])
    batch.sort(key = lambda x: -len(x[2]))
    _, bx = batchify(None, [x[2] for x in batch], eos = True)
    mask = maskset(bx)
    enc_out = model.enc(bx, mask)
    dec_in = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
    model.dec.hidden = model.enc.hidden
    if model.dec.feed_input:
        model.dec.attn.h = zeros(BATCH_SIZE, 1, HIDDEN_SIZE)
    heatmap = [[[""] + x[1] + [EOS]] for x in batch[:len(eos)]]
    while sum(eos) < len(eos) and t < MAX_LEN:
        dec_out = model.dec(dec_in, enc_out, t, mask)
        if BEAM_SIZE == 1:
            dec_in = greedy_search(model.dec, dec_out, tgt_vocab, batch, eos, heatmap)
        else:
            dec_in = beam_search(model.dec, dec_out, tgt_vocab, batch, eos, heatmap, t)
        t += 1
    batch, heatmap = zip(*sorted(zip(batch, heatmap), key = lambda x: (x[0][0], -x[0][4])))
    if VERBOSE >= 1:
        for i in range(len(heatmap)):
            if VERBOSE < 2 and i % BEAM_SIZE:
                continue
            print("heatmap[%d] =" % i)
            print(mat2csv(heatmap[i], rh = True))
    batch = [x for i, x in enumerate(batch) if not i % BEAM_SIZE]
    return [(x[1], [tgt_vocab[x] for x in x[3][:-1]], x[4].item()) for x in batch]

def predict(filename, model, x_cti, x_wti, y_wti):
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
        return run_model(model, data)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src.char_to_idx vocab.src.word_to_idx vocab.tgt.word_to_idx test_data" % sys.argv[0])
    for x, y, p in predict(sys.argv[5], *load_model()):
        print((x, y))
