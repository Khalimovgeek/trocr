from paddleocr import PaddleOCR
import cv2, re
import pytesseract
from utils import imagePlot,Utils
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='devanagari',
)

image_path = 'images/testml5.jpg'

results = ocr.ocr(image_path)

image = cv2.imread(image_path)
texts = []
for result in results[0]:

    box = result[0]

    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]

    x_min = int(min(x_coords))
    x_max = int(max(x_coords))

    y_min = int(min(y_coords))
    y_max = int(max(y_coords))

    cropped = image[y_min:y_max, x_min:x_max]
#     deionised = Utils.bilateral_filter(Utils.gray_scale(cropped))
#     # clahe = Utils.clahe(deionised, clip_limit=1.5, tile_grid_size=(6,6))
#     # threshold = Utils.threshold(clahe, max_value=255, block_size=15, C=4)
#     upscaled = cv2.resize(
#     deionised,
#     None,
#     fx=2,
#     fy=2,
#     interpolation=cv2.INTER_CUBIC
# )
#     threshold = Utils.threshold(upscaled, max_value=255, block_size=15, C=4)

#     # imagePlot.plot_images([cropped, deionised, upscaled, threshold], titles=["Cropped", "Deionised", "Upscaled", "Threshold"])
#     imagePlot.show_image(threshold, title="Thresholded Image")
#     # text = pytesseract.image_to_string(
#     #     cropped,
#     #     lang='mal',
#     #     config='--psm 6'
#     # )

#     # print(text)
#     # break
    gray = Utils.gray_scale(cropped)
    # deionised = Utils.bilateral_filter(gray)
    
    # upscaled = cv2.resize(
    #     deionised,
    #     None,
    #     fx=2,
    #     fy=2,
    #     interpolation=cv2.INTER_CUBIC
    # )
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 2. Upscale first (gives thresholding more pixels to work with)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(
        upscaled,
        (3,3),
        0
    )

    threshold = Utils.threshold(
        blurred,
        max_value=255,
        block_size=15,
        C=3
    )
    threshold = Utils.threshold(
        threshold,
        max_value=255,
        block_size=15,
        C=3
    )
    # kernel = cv2.getStructuringElement(
    #     cv2.MORPH_ELLIPSE,
    #     (2,2)
    # )

    # closed = cv2.morphologyEx(
    #     threshold,
    #     cv2.MORPH_CLOSE,
    #     kernel
    # )

    # smoothed = cv2.morphologyEx(
    #     closed,
    #     cv2.MORPH_OPEN,
    #     kernel
    # )

    # imagePlot.show_image(smoothed, title="Smoothed Image")
    # imagePlot.plot_images([cropped, gray, deionised, upscaled, blurred, threshold, closed, smoothed], titles=["Cropped", "Gray", "Deionised", "Upscaled", "Blurred", "Threshold", "Closed", "Smoothed"])
    text = pytesseract.image_to_string(threshold, lang='mal', config='--psm 11')
    text = text.replace('\x0c', '')
    text = text.replace('\n', ' ')

    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    if text:
        texts.append(text)
print(texts)

with open('output.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(text + ' ')
