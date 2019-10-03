from model import *
from utils import *
from evaluate import *

def load_data():
    bxc = [] # source character sequence batch
    bxw = [] # source word sequence batch
    by = [] # target sequence batch
    data = []
    cti = load_tkn_to_idx(sys.argv[2])
    wti = load_tkn_to_idx(sys.argv[3])
    print("loading %s..." % sys.argv[4])
    fo = open(sys.argv[4], "r")
    for line in fo:
        x, y = line.strip().split("\t")
        x = [i.split(":") for i in x.split(" ")]
        y = [int(i) for i in y.split(" ")]
        xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
        bxc.append(xc)
        bxw.append(xw)
        by.append(y)
        if len(by) == BATCH_SIZE:
            bxc, bxw = batchify(bxc, bxw, eos = True)
            _, by = batchify(None, by)
            data.append((bxc, bxw, by))
            bxc = []
            bxw = []
            by = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, cti, wti

def train():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[-1])
    data, cti, wti = load_data()
    model = ptrnet(len(cti), len(wti))
    enc_optim = torch.optim.Adam(model.enc.parameters(), lr = LEARNING_RATE)
    dec_optim = torch.optim.Adam(model.dec.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for xc, xw, y in data:
            loss = model(xc, xw, y) # forward pass and compute loss
            loss.backward() # compute gradients
            enc_optim.step() # update encoder parameters
            dec_optim.step() # update decoder parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(data)
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
    train()
