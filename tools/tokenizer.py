import sys
import re

RE_NAN = re.compile("([^ A-Za-z0-9\u4E00-\u9FFF\uAC00-\uD7AF])") # non-alphanumeric
RE_SPACE = re.compile("[\s\u3000]+") # whitespace
RE_APOS_L = re.compile("' (cause|d|em|ll|m|s|t|re|ve)\\b", re.IGNORECASE)
RE_APOS_R = re.compile("\\b(goin) '", re.IGNORECASE)

def tokenize(lang, filename):
    fo = open(filename)
    for line in fo:
        line = RE_NAN.sub(" \\1 ", line)
        line = RE_SPACE.sub(" ", line)
        line = RE_APOS_L.sub("'\\1", line)
        line = RE_APOS_R.sub("\\1'", line)
        line = line.strip()
        print(line)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: %s lang text" % sys.argv[0])
    tokenize(*sys.argv[1:])
