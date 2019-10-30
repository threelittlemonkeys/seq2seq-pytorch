from utils import *

def load_data():
    data = []
    cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    fo = open(sys.argv[1])
    if HRE:
        tmp = []
        txt = fo.read().strip().split("\n\n")
        for doc in txt:
            data.append([])
            for line in doc.split("\n"):
                x, y = load_line(line, cti, wti)
                if x and y:
                    data[-1].append((x, y))
        for doc in sorted(data, key = lambda x: -len(x)):
            tmp.extend(doc)
            tmp.append(None)
        data = tmp[:-1]
    else:
        for line in fo:
            line = line.strip()
            x, y = load_line(line, cti, wti)
            data.append((x, y))
        data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    fo.close()
    return data, cti, wti

def load_line(line, cti, wti):
    x, y = line.split("\t")
    x = tokenize(x, UNIT)
    y = [str(int(y) + 1)] if HRE else [str(int(x) + 1) for x in y.split(" ")] # PAD_IDX = 0
    if len(x) < MIN_LEN or len(x) > MAX_LEN:
        return None, None
    for w in x:
        for c in w:
            if c not in cti:
                cti[c] = len(cti)
        if w not in wti:
            wti[w] = len(wti)
    x = ["+".join(str(cti[c]) for c in w) + ":%d" % wti[w] for w in x]
    return x, y

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, cti, wti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
    save_tkn_to_idx(sys.argv[1] + ".word_to_idx", wti)
