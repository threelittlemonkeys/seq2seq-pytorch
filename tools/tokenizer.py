import sys
import re

RE_X_SPACE = re.compile("[\s\u3000]+") # whitespace
RE_X_NON_ALNUM = re.compile("([^ a-z0-9\u4E00-\u9FFF\uAC00-\uD7AF])", re.I) # non-alphanumeric

RE_EN_APOS_L = re.compile("(^| )' (cause|d|em|ll|m|s|t|re|ve)( |$)", re.I)
RE_EN_APOS_R = re.compile("(^| )(goin) '( |$)", re.I)
RE_EN_NOT = re.compile("(^| )(ca|could|did|does|do|had|has|have|is|was|wo|would)n 't( |$)", re.I)

def tokenize(lang, filename):
    fo = open(filename)
    for line in fo:

        line = RE_X_SPACE.sub(" ", line)
        line = RE_X_NON_ALNUM.sub(" \\1 ", line)

        line = RE_EN_APOS_L.sub(r"\1'\2\3", line)
        line = RE_EN_APOS_R.sub(r"\1\2'\3", line)
        line = RE_EN_NOT.sub(r"\1\2n't\3", line)

        line = line.strip()
        print(line)

    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: %s lang text" % sys.argv[0])
    tokenize(*sys.argv[1:])
