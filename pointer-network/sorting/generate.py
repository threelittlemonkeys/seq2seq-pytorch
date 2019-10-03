import sys
import random

if __name__ == "__main__": # generate lists of random numbers
    if len(sys.argv) != 4:
        sys.exit("Usage: %s max_num max_len data_size" % sys.argv[0])
    i = 1
    max_num = int(sys.argv[1]) # maximum number
    max_len = int(sys.argv[2]) # maximum sequence lengh
    data_size = int(sys.argv[3]) # data size
    pl = dict()
    ls = range(0, max_num + 1)
    while True:
        x = tuple(random.sample(ls, random.randint(1, max_len)))
        if x in pl:
            continue
        pl[x] = True
        y = sorted(range(len(x)), key = lambda i: x[i])
        print("%s\t%s" % (" ".join(map(str, x)), " ".join(map(str, y))))
        if i == data_size:
            break
        i += 1
