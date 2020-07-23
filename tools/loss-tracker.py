import sys

def loss_diff():

    fo_data = open(sys.argv[1])
    data = fo_data.read().strip().split("\n")
    fo_data.close()
    
    corpus = dict()
    
    fo_idx = open(sys.argv[2])
    fo_loss1 = open(sys.argv[3]) 
    fo_loss2 = open(sys.argv[4]) 
    
    for idx, loss1, loss2 in zip(fo_idx, fo_loss1, fo_loss2):
        idx = int(idx.strip())
        txt = data[idx]
        loss1 = float(loss1.strip())
        loss2 = float(loss2.strip())
        corpus[txt] = (loss1, loss2, loss2 / loss1)
    
    fo_idx.close()
    fo_loss1.close()
    fo_loss2.close()
    
    for seq, loss in sorted(corpus.items(), key = lambda x: x[1][2]):
        print("%.6f\t%.6f\t%.6f\t%s" % (*loss, seq))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s data idx loss1 loss2" % sys.argv[0])
    loss_diff()
