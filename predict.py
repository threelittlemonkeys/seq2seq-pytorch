import sys
import re
from model import *
from utils import *

def load_model():
    vocab_src = load_vocab(sys.argv[2], "src")
    vocab_tgt = load_vocab(sys.argv[3], "tgt")
    vocab_tgt = [word for word, _ in sorted(vocab_tgt.items(), key = lambda x: x[1])]
    enc = encoder(len(vocab_src))
    dec = decoder(len(vocab_tgt))
    print(enc)
    print(dec)
    load_checkpoint(sys.argv[1], enc, dec)
    return enc, dec, vocab_src, vocab_tgt

def run_model(enc, dec, vocab_tgt, data):
    line = []
    pred = []
    batch = []
    while len(data) < BATCH_SIZE:
        data.append(("", [EOS_IDX]))
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    for x, y in data:
        line.append(x)
        batch.append(y + [PAD_IDX] * (batch_len - len(y)))
    batch = Var(LongTensor(batch))
    enc_out = enc(batch)
    dec_in = Var(LongTensor([SOS_IDX] * BATCH_SIZE)).unsqueeze(1)
    dec_hidden = enc.hidden
    for i in range(batch.size(1)):
        dec_out, dec_hidden = dec(dec_in, dec_hidden)
        dec_in = Var(dec_out.data.topk(1)[1])
        print(vocab_tgt[scalar(dec_in)])
        if scalar(dec_in) == EOS_IDX:
            break

def predict():
    data = []
    enc, dec, vocab_src, vocab_tgt = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        tokens = tokenize(line)
        data.append((line, [vocab_src[i] for i in tokens] + [EOS_IDX]))
        if len(data) == BATCH_SIZE:
            run_model(enc, dec, vocab_tgt, data)
            data = []
    fo.close()
    if len(data):
        run_model(enc, dec, vocab_tgt, data)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
