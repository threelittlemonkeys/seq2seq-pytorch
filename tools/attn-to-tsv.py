import sys
import re

def split(fo, sep, z = 1024):
    buf, txt = fo.read(z), ""
    while buf:
        txt += buf
        i = 0
        j = txt.find(sep, i)
        while j != -1:
            yield txt[i:j]
            i = j + len(sep)
            j = txt.find(sep, i)
        buf, txt = fo.read(z), txt[i:]
    if txt:
        yield(txt)

def attn_to_tsv(filename, num = 0):
    fo = open(filename)
    idx = 0
    num = int(num)
    for block in split(fo, "\n\n"):
        if not re.match("attn\[[0-9]+\] =(\n\S*(\t\S)+)", block):
            continue
        if idx:
            print()
        block = block.split("\n")[1:]
        print("%d\t" % idx + "\n\t".join(block))
        idx += 1
        if idx == num:
            break
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: %s filename [number]" % sys.argv[0])
    attn_to_tsv(*sys.argv[1:])
