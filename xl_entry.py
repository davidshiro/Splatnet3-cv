from openpyxl import load_workbook

wb = load_workbook(filename = "template.xlsm", keep_vba=True)

ws = wb.active

ws['B5'] = "Comet Task Force"

wb.save("output.xlsm")