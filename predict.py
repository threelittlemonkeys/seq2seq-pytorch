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
    batch = []
    z = len(data)
    eos = [0 for _ in range(z)] # number of EOS tokens in the batch
    while len(data) < BATCH_SIZE:
        data.append(["", [EOS_IDX], []])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = Var(LongTensor([x[1] + [PAD_IDX] * (batch_len - len(x[1])) for x in data]))
    x_mask = batch.data.gt(0)
    enc_out = enc(batch, x_mask)
    dec_in = Var(LongTensor([SOS_IDX] * BATCH_SIZE)).unsqueeze(1)
    dec.hidden = enc.hidden
    dec.attn.hidden = Var(zeros(BATCH_SIZE, 1, HIDDEN_SIZE)) # for input feeding
    while sum(eos) < z:
        dec_out = dec(dec_in, enc_out, x_mask)
        dec_in = Var(dec_out.data.topk(1)[1])
        y = dec_in.view(-1).data.tolist()
        for i in range(z):
            if eos[i]:
                continue
            data[i][2].append(vocab_tgt[y[i]])
            if y[i] == EOS_IDX:
                eos[i] = 1
    return data[:z]

def predict():
    data = []
    enc, dec, vocab_src, vocab_tgt = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line = line.strip()
        tokens = tokenize(line, "word")
        data.append([line, [vocab_src[i] for i in tokens] + [EOS_IDX], []])
        if len(data) == BATCH_SIZE:
            result = run_model(enc, dec, vocab_tgt, data)
            for x in result:
                print(x)
            data = []
    fo.close()
    if len(data):
        result = run_model(enc, dec, vocab_tgt, data)
        for x in result:
            print(x)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
