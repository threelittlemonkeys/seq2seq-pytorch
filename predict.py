import sys
import re
from model import *
from utils import *

def load_model():
    vocab_src = load_vocab(sys.argv[2], "src")
    vocab_tgt = load_vocab(sys.argv[3], "tgt")
    vocab_tgt = [word for word, _ in sorted(vocab_tgt.items(), key = lambda x: x[1])]
    encoder = rnn_encoder(len(vocab_src))
    decoder = rnn_decoder(len(vocab_tgt))
    if CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    print(encoder)
    print(decoder)
    load_checkpoint(sys.argv[1], encoder, decoder)
    return encoder, decoder, vocab_src, vocab_tgt

def run_model(encoder, decoder, vocab_tgt, data):
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
    for i in range(batch.size(1)):
        z = encoder(batch[:, i].unsqueeze(1))
    decoder.hidden = encoder.hidden
    decoder_input = Var(LongTensor([SOS_IDX] * BATCH_SIZE)).unsqueeze(1)
    for i in range(batch.size(1)):
        decoder_output = decoder(decoder_input)
        decoder_input = Var(decoder_output.data.topk(1)[1])
        print(vocab_tgt[scalar(decoder_input)])
        if scalar(decoder_input) == EOS_IDX:
            break

def predict():
    data = []
    encoder, decoder, vocab_src, vocab_tgt = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        tokens = tokenize(line)
        data.append((line, [vocab_src[i] for i in tokens] + [vocab_src[EOS]]))
        if len(data) == BATCH_SIZE:
            run_model(encoder, decoder, vocab_tgt, data)
            data = []
    fo.close()
    if len(data):
        run_model(encoder, decoder, vocab_tgt, data)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab.src vocab.tgt test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
