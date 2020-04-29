from parameters import *

def greedy_search(dec, yo, batch, itw, eos, lens):
    p, yi = yo.topk(1)
    y = yi.view(-1).tolist()
    for i, _ in filter(lambda x: not x[1], enumerate(eos)):
        eos[i] = (y[i] == EOS_IDX)
        batch.y1[i].append(itw[y[i]])
        batch.prob[i] += p[i]
        batch.attn[i].append([itw[y[i]], *dec.attn.w[i, 0, :lens[i]]])
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
