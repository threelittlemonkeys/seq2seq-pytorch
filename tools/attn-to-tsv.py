import sys
import re

def attn_to_tsv():
    fo = open(sys.argv[1])
    data = fo.read().strip().split("\n\n")
    fo.close()
    
    idx = 0
    for block in data:
        if not re.match("attn\[[0-9]+\] =(\n\S*(\t\S)+)", block):
            continue
        block = block.split("\n")[1:]
        if idx: print()
        print("%d\t" % idx + "\n\t".join(block))
        idx += 1
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s filename" % sys.argv[0])
    attn_to_tsv()
