import sys

if __name__ == "__main__": # convert IOB tags to indices
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    fi = open(sys.argv[1])
    fo = open(sys.argv[1] + ".iob_to_idx", "w")
    for line in fi:
        line = line.strip().split(" ")
        x = [w[:-2] for w in line]
        y = [str(i) for i, j in enumerate(line) if j[-1] == "B"]
        fo.write("%s\t%s\n" % (" ".join(x), " ".join(y)))
    fi.close()
    fo.close()
