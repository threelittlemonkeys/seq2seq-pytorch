import sys

def load_data(corpus, raw_data, tkn_data, fn_idx, fn_loss, model_idx):
    fo_idx = open(fn_idx)
    fo_loss = open(fn_loss)

    for idx, loss in zip(fo_idx, fo_loss):
        idx = int(idx.strip())
        txt = raw_data[idx]
        loss = float(loss.strip())
        if txt not in corpus:
            src, tgt = tkn_data[idx].split("\t")
            src_len = src.count(" ") + 1
            tgt_len = tgt.count(" ") + 1
            corpus[txt] = [[-1, -1], (src_len, tgt_len)]
        corpus[txt][0][model_idx] = loss

    fo_idx.close()
    fo_loss.close()

def dcce(m1_loss, m2_loss, src_len, tgt_len): # dual conditional cross-entropy
    if m1_loss < 0 or m2_loss < 0:
        return -1
    return (m1_loss + m2_loss) / 2 + abs(m1_loss - m2_loss)

def loss_filter():
    corpus = dict()

    with open(sys.argv[1]) as fo:
        raw_data = fo.read().strip().split("\n")
    with open(sys.argv[2]) as fo:
        tkn_data = fo.read().strip().split("\n")

    load_data(corpus, raw_data, tkn_data, *sys.argv[3:5], 0) # model 1
    load_data(corpus, raw_data, tkn_data, *sys.argv[5:7], 1) # model 2

    for txt, (loss, lens) in corpus.items():
        score = dcce(*loss, *lens)
        corpus[txt].append(score)

    for txt, (loss, lens, score) in sorted(corpus.items(), key = lambda x: -x[1][-1]):
        if score < 0:
            continue
        print("%.6f\t%.6f\t%d\t%d\t%.6f\t%s" % (*loss, *lens, score, txt))

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s raw_data tkn_data m1_idx m1_loss m2_idx m2_loss" % sys.argv[0])
    loss_filter()
