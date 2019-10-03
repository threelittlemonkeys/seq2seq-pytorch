from utils import *

def load_data():
    data = []
    cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = tokenize(y, UNIT)
        if len(x) < MIN_LEN or len(x) > MAX_LEN:
            continue
        for w in x:
            for c in w:
                if c not in cti:
                    cti[c] = len(cti)
            if w not in wti:
                wti[w] = len(wti)
        x = ["+".join(str(cti[c]) for c in w) + ":%d" % wti[w] for w in x]
        y = [str(int(w) + 1) for w in y] + [str(len(y) + 1)]
        data.append((x, y))
    fo.close()
    data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    return data, cti, wti

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, cti, wti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
    save_tkn_to_idx(sys.argv[1] + ".word_to_idx", wti)

