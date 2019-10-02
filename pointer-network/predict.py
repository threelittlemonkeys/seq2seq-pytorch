from model import *
from utils import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2])
    wti = load_tkn_to_idx(sys.argv[3])
    model = ptrnet(len(cti), len(wti))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti

def greedy_search(dec, dec_out, batch, eos, heatmap):
    p, yw = dec_out.topk(1)
    y = yw.view(-1).tolist()
    for i in range(len(eos)):
        if eos[i]:
            continue
        eos[i] = (y[i] == len(batch[i][1]) or y[i] in batch[i][5])
        batch[i][5].append(y[i])
        batch[i][6] += p[i]
        heatmap[i].append([y[i]] + dec.attn.a[i].tolist())
    return yw 

def beam_search(dec, dec_out, batch, eos, heatmap, t):
    bp, by = dec_out[:len(eos)].topk(BEAM_SIZE) # [B * BEAM_SIZE, BEAM_SIZE]
    bp += Tensor([-10000 if b else a[6] for a, b in zip(batch, eos)]).unsqueeze(1)
    bp = bp.view(-1, BEAM_SIZE ** 2) # [B, BEAM_SIZE * BEAM_SIZE]
    by = by.view(-1, BEAM_SIZE ** 2)
    if t == 0: # remove non-first duplicate beams
        bp = bp[:, :BEAM_SIZE]
        by = by[:, :BEAM_SIZE]
    for i, (p, y) in enumerate(zip(bp, by)): # for each sequence
        y = y.tolist()
        j = i * BEAM_SIZE
        b1, m1 = [], [] # batch and heatmap to be updated
        if VERBOSE >= 2:
            for k in range(0, len(p), BEAM_SIZE): # for each beam
                q = j + k // BEAM_SIZE
                w = [(next(reversed(batch[q][5]), SOS_IDX), batch[q][6])] # previous token
                w += list(zip(y, p))[k:k + BEAM_SIZE] # current candidates
                w = [(a, round(b.item(), 4)) for a, b in w]
                print("beam[%d][%d][%d] =" % (i, t, q), w[0], "->", w[1:])
        for p, k in zip(*p.topk(BEAM_SIZE)): # for each n-best candidate
            q = j + k // BEAM_SIZE
            b1.append(batch[q].copy())
            b1[-1][5] = b1[-1][5] + [y[k]] # token
            b1[-1][6] = p # probability
            m1.append(heatmap[q].copy())
            m1[-1].append([y[k]] + dec.attn.a[q][:len(batch[j][1]) + 1].tolist())
        for k in filter(lambda x: eos[j + x], range(BEAM_SIZE)): # append completed sequences
            b1.append(batch[j + k])
            m1.append(heatmap[j + k])
        topk = sorted(zip(b1, m1), key = lambda x: -x[0][6])[:BEAM_SIZE]
        for k, (b1, m1) in enumerate(topk, j):
            eos[k] = (b1[5][-1] == len(b1[1]) or b1[5][-1] in b1[5][:-1])
            batch[k] = b1
            heatmap[k] = m1
            if VERBOSE >= 2:
                print("output[%d][%d][%d] = " % (i, t, k), end = "")
                print((b1[5], round(b1[6].item(), 4)))
        if VERBOSE >= 2:
            print()
    return LongTensor([next(reversed(x[5]), SOS_IDX) for x in batch]).unsqueeze(1)

def run_model(model, batch):
    t = 0
    eos = [False for _ in batch] # number of completed sequences in the batch
    while len(batch) < BATCH_SIZE:
        batch.append([-1, [], [[EOS_IDX]], [EOS_IDX], [], [], 0])
    batch.sort(key = lambda x: -len(x[3]))
    xc, xw = batchify(*zip(*[(x[2], x[3]) for x in batch]), eos = True)
    mask = maskset(xw)
    enc_out = model.enc(xc, xw, mask)
    yc = LongTensor([[SOS_IDX]] * BATCH_SIZE)
    yw = LongTensor([SOS_IDX] * BATCH_SIZE).unsqueeze(1)
    model.dec.hidden = model.enc.hidden
    heatmap = [[[""] + x[2] + [EOS]] for x in batch[:len(eos)]]
    while sum(eos) < len(eos) and t < MAX_LEN:
        dec_out = model.dec(yc.unsqueeze(1), yw, enc_out, t, mask)
        if BEAM_SIZE == 1:
            yw = greedy_search(model.dec, dec_out, batch, eos, heatmap)
        else:
            yw = beam_search(model.dec, dec_out, batch, eos, heatmap, t)
        yc = torch.cat([xc[i, j] for i, j in enumerate(yw)])
        t += 1
    batch, heatmap = zip(*sorted(zip(batch, heatmap), key = lambda x: (x[0][0], -x[0][6])))
    if VERBOSE >= 1:
        for i in range(len(heatmap)):
            if VERBOSE < 2 and i % BEAM_SIZE:
                continue
            print("heatmap[%d] =" % i)
            print(heatmap[i])
            print(mat2csv(heatmap[i], rh = True))
    batch = [x for i, x in enumerate(batch) if not i % BEAM_SIZE]
    return [(x[1], x[4], x[5][:-1]) for x in batch]

def predict(filename, model, cti, wti):
    data = []
    result = []
    fo = open(filename)
    for idx, line in enumerate(fo):
        line = line.strip()
        if re.match("[^\t]+\t[0-9]+( [0-9]+)*$", line):
            line, y = line.split("\t")
            y = [int(x) for x in y.split(" ")]
        else: # no ground truth provided
            y = []
        x = tokenize(line, UNIT)
        xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x]
        xw = [wti[w] if w in wti else UNK_IDX for w in x]
        data.extend([[idx, x, xc, xw, y, [], Tensor([0])] for _ in range(BEAM_SIZE)])
    fo.close()
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            for y in run_model(model, batch):
                yield y

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model char_to_idx word_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    result = predict(sys.argv[4], *load_model())
    for x, y0, y1 in result:
        print((x, y0, y1) if y0 else (x, y1))
