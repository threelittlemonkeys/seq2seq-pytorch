from model import *
from utils import *
from dataloader import *
from beamsearch import *

def load_model():

    x_cti = load_tkn_to_idx(sys.argv[2])
    x_wti = load_tkn_to_idx(sys.argv[3])
    y_wti = load_tkn_to_idx(sys.argv[4])
    y_itw = load_idx_to_tkn(sys.argv[4])

    model = rnn_encoder_decoder(x_cti, x_wti, y_wti)
    print(model)

    load_checkpoint(sys.argv[1], model)

    return model, x_cti, x_wti, y_itw

def run_model(model, data, itw):

    with torch.no_grad():
        model.eval()

        for batch in data.split():

            xc, xw, _, lens = batch.sort()
            xc, xw = data.tensor(xc, xw, lens, eos = True)
            b, t = len(xw), 0 # batch size, time step
            eos = [False for _ in xw] # EOS states
            mask, lens = maskset(xw)
            model.dec.M, model.dec.H = model.enc(b, xc, xw, lens)
            model.dec.h = zeros(b, 1, HIDDEN_SIZE)
            if ATTN:
                model.dec.attn.V = zeros(b, 1, HIDDEN_SIZE)
            if COPY:
                model.dec.copy.V = zeros(b, 1, HIDDEN_SIZE)
            yi = LongTensor([[SOS_IDX]] * b)
            batch.y1 = [[] for _ in range(b)]
            batch.prob = [Tensor([0]) for _ in range(b)]
            batch.attn = [[["", *batch.x1[i], EOS]] for i in batch.idx]

            while t < MAX_LEN and sum(eos) < len(eos):
                yo = model.dec(xw, yi, mask)
                args = (model.dec, batch, itw, eos, lens, yo)
                yi = beam_search(*args, t) if BEAM_SIZE > 1 else greedy_search(*args)
                t += 1
            batch.unsort()

            if VERBOSE:
                print()
                for i, x in filter(lambda x: not x[0] % BEAM_SIZE, enumerate(batch.attn)):
                    print("attn[%d] =" % (i // BEAM_SIZE))
                    print(mat2csv(x, rh = True))

            for i, (x0, y0, y1) in enumerate(zip(batch.x0, batch.y0, batch.y1)):
                if not i % BEAM_SIZE: # use the best candidate from each beam
                    y1 = [itw[y] for y in y1[:-1]]
                    yield x0, y0, y1

def predict(filename, model, x_cti, x_wti, y_itw):

    data = dataloader()
    fo = open(filename)

    for x0 in fo:
        x0 = x0.strip()
        y0 = None
        if x0.count("\t") == 1:
            x0, y0 = x0.split("\t")
        x1 = tokenize(x0, UNIT)
        xc = [[x_cti.get(c, UNK_IDX) for c in w] for w in x1]
        xw = [x_wti.get(w, UNK_IDX) for w in x1]
        data.append_item(x0, x1, xc, xw, y0)
        for _ in range(BEAM_SIZE - 1):
            data.append_row()
            data.append_item(x0, x1, xc, xw, y0)
        data.append_row()
    fo.close()
    data.strip()

    return run_model(model, data, y_itw)

if __name__ == "__main__":

    if len(sys.argv) != 6:
        sys.exit("Usage: %s model vocab.src.char_to_idx vocab.src.word_to_idx vocab.tgt.word_to_idx test_data" % sys.argv[0])

    for x, y0, y1 in predict(sys.argv[5], *load_model()):
        print((x, y0, y1) if y0 else (x, y1))
