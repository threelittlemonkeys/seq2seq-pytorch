import re
from model import *

def tokenize(s):
    s = s.lower()
    s = re.sub("[^ a-z0-9" + chr(0xAC00) + "-" + chr(0xD7A3) + "]+", "", s)
    s = re.sub("\s+", " ", s)
    s = re.sub("^ | $", "", s)
    return s.split(" ")

def load_vocab(filename, ext):
    print("loading vocab.%s..." % ext)
    vocab = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        vocab[line] = len(vocab)
    fo.close()
    return vocab

def load_checkpoint(filename, encoder = None, decoder = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if encoder and decoder:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, encoder, decoder, epoch, loss):
    print("saving model...")
    checkpoint = {}
    checkpoint["encoder_state_dict"] = encoder.state_dict()
    checkpoint["decoder_state_dict"] = decoder.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    torch.save(checkpoint, filename + ".epoch%d" % epoch)
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
