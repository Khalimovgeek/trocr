from mmocr.apis import MMOCRInferencer

ocr = MMOCRInferencer(det='DB_r18', rec='CRNN')

result = ocr('images/testml (3).jpg', show=True)

print(result['predictions'])