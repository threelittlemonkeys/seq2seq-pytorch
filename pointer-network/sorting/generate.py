import sys
import random

# generate lists of random numbers

if len(sys.argv) != 6:
    sys.exit("Usage: %s unit max_num max_width max_height data_size" % sys.argv[0])

unit = sys.argv[1] # unit
max_num = int(sys.argv[2]) # maximum number
max_width = int(sys.argv[3]) # maximum sequence lengh
max_height = int(sys.argv[4]) # maximum sequence lengh
data_size = int(sys.argv[5]) # data size
data = range(0, max_num + 1)

def sorted_idxs(xs):
    return sorted(range(1, len(xs) + 1), key = lambda i: xs[i - 1])

if unit == "word":
    pl = set()
    while len(pl) < data_size:
        xs = tuple(random.sample(data, random.randint(1, max_width)))
        if xs in pl:
            continue
        pl.add(xs)
        ys = sorted_idxs(xs)
        print(" ".join(map(str, xs)), " ".join(map(str, ys)), sep = "\t")

if unit == "sent":
    for z0 in range(data_size):
        z1 = random.randint(1, max_height)
        xs = []
        while len(xs) < z1:
            x = random.sample(data, random.randint(1, max_width))
            if sum(x) not in map(sum, xs): # same sum not allowed
                xs.append(x)
        ys = sorted_idxs(list(map(sum, xs)))
        if z0:
            print()
        for x, y in zip(xs, ys):
            print(" ".join(map(str, x)), y, sep = "\t")
