from utils import *
from dataloader import *
from pointer_networks import *
from beamsearch import *

def load_model():

    cti = load_tkn_to_idx(sys.argv[2])
    wti = load_tkn_to_idx(sys.argv[3])

    model = pointer_networks(cti, wti)
    print(model)

    load_checkpoint(sys.argv[1], model)

    return model, cti, wti

def run_model(model, data):

    with torch.no_grad():
        model.eval()

        for batch in data.batchify(BATCH_SIZE * BEAM_SIZE):

            xc, xw, lens = batch.xc, batch.xw, batch.lens
            xc, xw = data.to_tensor(xc, xw, lens, eos = True)
            eos = [False for _ in lens] # EOS states
            b, t = len(lens), 0
            mask, lens = maskset(
                Tensor([[i >= j for j in range(max(lens) + 1)] for i in lens])
                if HRE else xw
            )

            xh, model.dec.M, model.dec.H = model.enc(xc, xw, lens)
            model.init_state(b)
            yc = LongTensor([[[SOS_IDX]]] * b)
            yw = LongTensor([[SOS_IDX]] * b)
            yi = model.enc.embed(b, yc, yw)

            batch.y1 = [[] for _ in range(b)]
            batch.prob = [0 for _ in range(b)]
            batch.attn = [[["", *batch.x0[i], EOS]] for i in range(b)]

            while t < lens[0] and sum(eos) < len(eos):
                yo = model.dec(yi, mask)
                args = (model.dec, batch, eos, lens, yo)
                y1 = beam_search(*args, t) if BEAM_SIZE > 1 else greedy_search(*args)
                yi = torch.cat([xh[i, j] for i, j in enumerate(y1)]).unsqueeze(1)
                t += 1

            if VERBOSE:
                for i in range(0, len(batch.y1), BEAM_SIZE):
                    i //= BEAM_SIZE
                    print("attn[%d] =" % i)
                    print(mat2csv(batch.attn[i]), end = "\n\n")

            for i, (x0, y0, y1) in enumerate(zip(batch.x0, batch.y0, batch.y1)):
                if not i % BEAM_SIZE: # use the best candidate from each beam
                    y1 = [y + 1 for y in y1[:-1]]
                    yield x0, y0, y1

def predict(model, cti, wti, filename):

    data = dataloader(batch_first = True, hre = HRE)

    with open(filename) as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))

    for block in text:
        data.append_row()

        for line in block.split("\n"):
            if re.match("[^\t]+\t[0-9]+( [0-9]+)*$", line):
                x0, y0 = line.split("\t")
                y0 = list(map(int, y0.split(" ")))
            else: # no ground truth provided
                x0, y0 = line.strip(), []
            x1 = tokenize(x0)

            xc = [[cti.get(c, UNK_IDX) for c in w] for w in x1]
            xw = [wti.get(w, UNK_IDX) for w in x1]

            data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)

        for _ in range(BEAM_SIZE - 1):
            data.clone_row()

    return run_model(model, data)

if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.exit("Usage: %s model char_to_idx word_to_idx test_data" % sys.argv[0])

    for x, y0, y1 in predict(*load_model(), sys.argv[4]):
        if y0:
            print((x, y0))
        print((x, y1))
