import sys
import re
import xlsxwriter

def attn_to_xslx():
    fo = open(sys.argv[1])
    data = fo.read().strip().split("\n\n")
    fo.close()

    workbook = xlsxwriter.Workbook(sys.argv[1] + ".attn.xlsx")
    worksheet = workbook.add_worksheet()

    row_id = 0
    sent_id = 0
    for block in data:
        if not re.match("attn\[[0-9]+\] =(\n\S*(\t\S)+)", block):
            continue
        if sent_id:
            row_id += 1
        worksheet.write(row_id, 0, sent_id)
        for i, row in enumerate(block.split("\n")[1:]):
            for col_id, txt in enumerate(row.split("\t"), 1):
                if i > 0 and col_id > 1:
                    txt = float(txt)
                    rgb = "#FF%s" % (("%X" % int(0x10 * (1 - txt) - 1e-6) * 4))
                    cell_format = workbook.add_format({"bg_color": rgb})
                    worksheet.write(row_id, col_id, txt, cell_format)
                else:
                    worksheet.write(row_id, col_id, txt)
            row_id += 1
        sent_id += 1

    workbook.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s filename" % sys.argv[0])
    attn_to_xslx()
