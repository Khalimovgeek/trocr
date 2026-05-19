import easyocr

reader = easyocr.Reader(['en'])

result = reader.readtext('images/testml.jpg')

for r in result:
    print(r[1])