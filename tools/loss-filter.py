import sys

def load_data(corpus, data, fn_idx, fn_loss, model_idx):
    fo_idx = open(fn_idx)
    fo_loss = open(fn_loss)

    for idx, loss in zip(fo_idx, fo_loss):
        idx = int(idx.strip())
        txt = data[idx]
        loss = float(loss.strip())
        if txt not in corpus:
            corpus[txt] = [-1, -1]
        corpus[txt][model_idx] = loss

    fo_idx.close()
    fo_loss.close()

def loss_tracker():
    corpus = dict()

    fo_data = open(sys.argv[1])
    data = fo_data.read().strip().split("\n")
    fo_data.close()

    load_data(corpus, data, *sys.argv[2:4], 0)
    load_data(corpus, data, *sys.argv[4:6], 1)

    for txt, loss in sorted(corpus.items()):
        score = abs(loss[0] - loss[1]) + sum(loss) / 2
        corpus[txt].append(score)

    for txt, (*loss, score) in sorted(corpus.items(), key = lambda x: -x[1][-1]):
        print("%.6f\t%.6f\t%.6f\t%s" % (*loss, score, txt))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s m1_data m1_idx m1_loss m2_idx m2_loss" % sys.argv[0])
    loss_tracker()
