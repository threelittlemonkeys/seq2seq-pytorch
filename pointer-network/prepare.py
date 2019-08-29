from utils import *

def load_data():
    data = []
    vocab = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        src, tgt = line.split("\t")
        src_tokens = tokenize(src, UNIT)
        tgt_tokens = tokenize(tgt, UNIT)
        if len(src_tokens) < MIN_LEN or len(src_tokens) > MAX_LEN:
            continue
        src_seq = []
        tgt_seq = []
        for w in src_tokens:
            if w not in vocab:
                vocab[w] = len(vocab)
            src_seq.append(str(vocab[w]))
        for w in tgt_tokens:
            tgt_seq.append(str(int(w) + 1))
        tgt_seq.append(str(len(src_seq) + 1))
        data.append((src_seq, tgt_seq))
    fo.close()
    data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    return data, vocab

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, vocab = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_vocab(sys.argv[1] + ".vocab", vocab)
