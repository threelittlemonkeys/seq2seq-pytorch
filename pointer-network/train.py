from utils import *
from dataloader import *
from pointer_networks import *
from evaluate import *

def load_data():

    data = dataloader(batch_first = True, hre = HRE)
    batch = []
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx

    print(f"loading {sys.argv[4]}")

    with open(sys.argv[4], "r") as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))

    for block in text:
        data.append_row()

        for line in block.split("\n"):
            x, y = line.split("\t")
            x = [x.split(":") for x in x.split(" ")]
            y = list(map(int, y.split(" "))) + ([] if HRE else [len(x) + 1])
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            data.append_item(xc = xc, xw = xw, y0 = y)

        if HRE:
            data.append_item(y0 = [len(data.y0[-1]) + 1])

    for _batch in data.split(BATCH_SIZE):
        xc, xw = data.tensor(_batch.xc, _batch.xw, _batch.lens, eos = True)
        _, y0 = data.tensor(None, _batch.y0)
        batch.append((xc, xw, y0))

    print("data size: %d" % (len(data.y0)))
    print("batch size: %d" % (BATCH_SIZE))

    return batch, cti, wti

def train():

    num_epochs = int(sys.argv[-1])
    batch, cti, wti = load_data()
    model = pointer_networks(cti, wti)
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

        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti]
            evaluate(predict(sys.argv[5], *args), True)
            model.train()
            print()

if __name__ == "__main__":

    if len(sys.argv) not in [6, 7]:
        sys.exit("Usage: %s model char_to_idx word_to_idx training_data (validation data) num_epoch" % sys.argv[0])
    if len(sys.argv) == 6:
        EVAL_EVERY = False

    train()
