import sys
import os.path
import re
import time
from model import *
from utils import *
from random import random

def load_data():
    data = []
    batch_src = []
    batch_tgt = []
    batch_len_src = 0
    batch_len_tgt = 0
    print("loading data...")
    vocab_src = load_vocab(sys.argv[2], "src")
    vocab_tgt = load_vocab(sys.argv[3], "tgt")
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = line.strip()
        words = [int(i) for i in line.split(" ")]
        len_src = words.pop()
        len_tgt = len(words) - len_src
        if len_src > batch_len_src:
            batch_len_src = len_src
        if len_tgt > batch_len_tgt:
            batch_len_tgt = len_tgt
        batch_src.append(words[:len_src] + [vocab_src[EOS]])
        batch_tgt.append(words[len_src:] + [vocab_tgt[EOS]])
        if len(batch_src) == BATCH_SIZE:
            for seq in batch_src:
                seq.extend([PAD_IDX] * (batch_len_src - len(seq) + 1))
            for seq in batch_tgt:
                seq.extend([PAD_IDX] * (batch_len_tgt - len(seq) + 1))
            data.append((Var(LongTensor(batch_src)), Var(LongTensor(batch_tgt))))
            batch_src = []
            batch_tgt = []
            batch_len_src = 0
            batch_len_tgt = 0
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, vocab_src, vocab_tgt

def train():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data, vocab_src, vocab_tgt = load_data()
    if VERBOSE:
        itow_src = [word for word, _ in sorted(vocab_src.items(), key = lambda x: x[1])]
        itow_tgt = [word for word, _ in sorted(vocab_tgt.items(), key = lambda x: x[1])]
    encoder = rnn_encoder(len(vocab_src))
    decoder = rnn_decoder_vanilla(len(vocab_tgt))
    encoder_optim = torch.optim.SGD(encoder.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    decoder_optim = torch.optim.SGD(decoder.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], encoder, decoder) if os.path.isfile(sys.argv[1]) else 0
    if CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(encoder)
    print(decoder)
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        for ii, (x, y) in enumerate(data):
            loss = 0
            encoder.zero_grad()
            decoder.zero_grad()
            if VERBOSE:
                pred = [[] for _ in range(BATCH_SIZE)]

            # encoder forward pass
            encoder_outputs = Var(zeros(BATCH_SIZE, x.size(1), HIDDEN_SIZE))
            for t in range(x.size(1)):
                encoder_outputs[:, t] = encoder(x[:, t].unsqueeze(1))

            # decoder forward pass
            decoder_input = Var(LongTensor([SOS_IDX] * BATCH_SIZE)).unsqueeze(1)
            decoder_outputs = Var(zeros(BATCH_SIZE, y.size(1), HIDDEN_SIZE))
            decoder.hidden = encoder.hidden

            # teacher forcing
            for t in range(y.size(1)):
                decoder_output = decoder(decoder_input)
                mask = Var(y[:, t].data.gt(0).float().unsqueeze(-1).expand_as(decoder_output))
                loss += F.nll_loss(decoder_output * mask, y[:, t])
                decoder_input = y[:, t].unsqueeze(1)
                if VERBOSE:
                    for i, j in enumerate(decoder_output.data.topk(1)[1]):
                        pred[i].append(scalar(Var(j)))

            loss.backward()
            encoder_optim.step()
            decoder_optim.step()
            loss = scalar(loss)
            loss_sum += loss
            if VERBOSE:
                print("epoch = %d, iteration = %d, loss = %f\n" % (ei, ii + 1, loss))
                for a, b, c in zip(x, y, pred):
                    print(" ".join([itow_src[scalar(i)] for i in a]))
                    print(" ".join([itow_tgt[i] for i in c][:len_unpadded(b)]))
                    print()
        if ei % SAVE_EVERY == 0 or ei == epoch + num_epochs:
            save_checkpoint(filename, encoder, decoder, ei, loss_sum / len(data))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src vocab.tgt training_data num_epoch" % sys.argv[0])
    train()
