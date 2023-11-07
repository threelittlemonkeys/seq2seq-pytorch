from utils import *
from dataloader import *
from transformer import *

def load_data():

    data = dataloader(batch_first = True)
    batch = []
    x_cti = load_tkn_to_idx(sys.argv[2]) # source char_to_idx
    x_wti = load_tkn_to_idx(sys.argv[3]) # source word_to_idx
    y_wti = load_tkn_to_idx(sys.argv[4]) # target word_to_idx

    print(f"loading {sys.argv[5]}")

    fo = open(sys.argv[5])
    for line in fo:
        x, y = line.strip().split("\t")
        x = [x.split(":") for x in x.split(" ")]
        y = list(map(int, y.split(" ")))
        xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
        data.append_row()
        data.append_item(xc = xc, xw = xw, y0 = y)
    fo.close()

    for _batch in data.batchify(BATCH_SIZE):
        xc, xw, y0, lens = _batch.sort()
        xc, xw = data.to_tensor(xc, xw, lens, eos = True)
        _, y0 = data.to_tensor(None, y0, sos = True, eos = True)
        batch.append((xc, xw, y0))

    print("data size: %d" % len(data.y0))
    print("batch size: %d" % (BATCH_SIZE))

    return batch, x_cti, x_wti, y_wti

def train():

    num_epochs = int(sys.argv[-1])
    batch, x_cti, x_wti, y_wti = load_data()
    model = transformer(x_cti, x_wti, y_wti)
    print(model)

    enc_optim = torch.optim.Adam(model.enc.parameters(), lr = LEARNING_RATE)
    dec_optim = torch.optim.Adam(model.dec.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])

    print("training model")

    for ei in range(epoch + 1, epoch + num_epochs + 1):

        loss_sum = 0
        timer = time()

        for xc, xw, y0 in batch:

            loss = model(xc, xw, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            enc_optim.step() # update encoder parameters
            dec_optim.step() # update decoder parameters
            loss_sum += loss.item()

        timer = time() - timer
        loss_sum /= len(batch)

        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

if __name__ == "__main__":

    if len(sys.argv) != 7:
        sys.exit("Usage: %s model vocab.src.char_to_idx vocab.src.word_to_idx vocab.tgt.word_to_idx training_data num_epoch" % sys.argv[0])

    train()
