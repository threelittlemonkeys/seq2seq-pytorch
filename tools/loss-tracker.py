import sys

def loss_tracker():

    corpus = dict()

    fo_data = open(sys.argv[1])
    fo_idx = open(sys.argv[2])
    fo_loss1 = open(sys.argv[3])
    fo_loss2 = open(sys.argv[4])
    data = fo_data.read().strip().split("\n")

    for idx, loss1, loss2 in zip(fo_idx, fo_loss1, fo_loss2):
        idx = int(idx.strip())
        txt = data[idx]
        loss1 = float(loss1.strip())
        loss2 = float(loss2.strip())
        corpus[txt] = (loss1, loss2, loss2 / loss1)

    fo_data.close()
    fo_idx.close()
    fo_loss1.close()
    fo_loss2.close()

    for txt, loss in sorted(corpus.items(), key = lambda x: -x[1][1]):
        print("%.6f\t%.6f\t%.6f\t%s" % (*loss, txt))

    # loss1_avrg = sum(x for x, *_ in corpus.values()) / len(corpus)
    # loss2_avrg = sum(x for _, x, _ in corpus.values()) / len(corpus)
    # ratio_avrg = sum(x for *_, x in corpus.values()) / len(corpus)
    # print("%.6f\t%.6f\t%.6f" % (loss1_avrg, loss2_avrg, ratio_avrg))

if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit("Usage: %s data idx loss1 loss2" % sys.argv[0])

    loss_tracker()
