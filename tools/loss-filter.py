import sys

def loss_tracker():
    corpus = dict()

    fo_data = open(sys.argv[1])
    fo_idx = open(sys.argv[2])
    fo_loss = open(sys.argv[3])
    data = fo_data.read().strip().split("\n")

    for idx, loss in zip(fo_idx, fo_loss):
        idx = int(idx.strip())
        txt = data[idx]
        loss = float(loss.strip())
        if txt not in corpus:
            corpus[txt] = [-1, -1]
        if loss > corpus[txt][0]:
            corpus[txt][0] = loss

    fo_data.close()
    fo_idx.close()
    fo_loss.close()

    fo_data = open(sys.argv[4])
    fo_idx = open(sys.argv[5])
    fo_loss = open(sys.argv[6])
    data = fo_data.read().strip().split("\n")

    for idx, loss in zip(fo_idx, fo_loss):
        idx = int(idx.strip())
        txt = "\t".join(data[idx].split("\t")[::-1])
        loss = float(loss.strip())
        if txt not in corpus:
            corpus[txt] = [-1, -1]
        if loss > corpus[txt][1]:
            corpus[txt][1] = loss

    fo_data.close()
    fo_idx.close()
    fo_loss.close()

    _corpus = dict()
    for txt, loss in sorted(corpus.items()):
        if loss[0] < 0 or loss[1] < 0:
            continue
        score = abs(loss[0] - loss[1]) + sum(loss) / 2
        _corpus[txt] = (*loss, score)
    corpus = _corpus

    for txt, (*loss, score) in sorted(corpus.items(), key = lambda x: -x[1][-1]):
        print("%.6f\t%.6f\t%.6f\t%s" % (*loss, score, txt))

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s m1_data m1_idx m1_loss m2_data m2_idx m2_loss" % sys.argv[0])
    loss_tracker()
