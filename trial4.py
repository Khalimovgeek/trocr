import keras_ocr
import matplotlib.pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

images = [
    keras_ocr.tools.read('images/testml.jpg')
]

prediction_groups = pipeline.recognize(images)

for text in prediction_groups[0]:
    print(text[0])