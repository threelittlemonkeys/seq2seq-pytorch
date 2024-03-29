import sys
import re
import xlsxwriter

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

def attn_to_xslx(filename, num = 0):

    fo = open(filename)
    num = int(num)
    workbook = xlsxwriter.Workbook(filename + ".attn.xlsx")
    worksheet = workbook.add_worksheet()

    sid, rid = 0, 0

    for block in split(fo, "\n\n"):

        if not re.match("attn\[[0-9]+\] =(\n\S*(\t\S)+)", block):
            continue

        if sid:
            rid += 1

        worksheet.write(rid, 0, sid)

        for i, row in enumerate(block.split("\n")[1:]):
            for cid, txt in enumerate(row.split("\t"), 1):
                if i > 0 and cid > 1:
                    txt = float(txt)
                    rgb = "#FF%s" % (("%02X" % int(0xFF * (1 - txt)) * 2))
                    cell_format = workbook.add_format({"bg_color": rgb})
                    worksheet.write(rid, cid, txt, cell_format)
                else:
                    worksheet.write(rid, cid, txt)
            rid += 1
        sid += 1

        if sid == num:
            break

    fo.close()
    workbook.close()

if __name__ == "__main__":

    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: %s filename [number]" % sys.argv[0])

    attn_to_xslx(*sys.argv[1:])
