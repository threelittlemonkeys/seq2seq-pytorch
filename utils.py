import re
from model import *

def tokenize(x, unit):
    x = x.lower()
    # x = re.sub("[^ a-z0-9\uAC00-\uD7A3]+", "", x)
    if unit == "char":
        x = re.sub("\s+", "", x)
        return list(x)
    elif unit == "word":
        x = re.sub("\s+", " ", x)
        x = re.sub("^ | $", "", x)
        return x.split(" ")

def load_vocab(filename, ext):
    print("loading vocab.%s..." % ext)
    vocab = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        vocab[line] = len(vocab)
    fo.close()
    return vocab

def load_checkpoint(filename, enc = None, dec = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if enc and dec:
        enc.load_state_dict(checkpoint["encoder_state_dict"])
        dec.load_state_dict(checkpoint["decoder_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, enc, dec, epoch, loss, timer):
    log = "epoch = %d, loss = %f, time = %f" % (epoch, loss, timer)
    if filename and enc and dec:
        print("saving model...")
        checkpoint = {}
        checkpoint["encoder_state_dict"] = enc.state_dict()
        checkpoint["decoder_state_dict"] = dec.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        log = "saved model: " + log
    print(log)
