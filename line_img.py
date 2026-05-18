import cv2
import numpy as np

def word_segmentation(line_img):
    # 1. grayscale
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

    # 2. binarize (adaptive is better for OCR)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # 3. projection profile (vertical sum)
    vertical_sum = np.sum(bin_img, axis=0)

    # 4. threshold to find gaps
    threshold = np.max(vertical_sum) * 0.05  # tweak this

    is_text = vertical_sum > threshold

    words = []
    start = None

    for i, val in enumerate(is_text):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            if end - start > 3:  # ignore noise
                words.append((start, end))
            start = None

    # last word edge case
    if start is not None:
        words.append((start, len(is_text)))

    # 5. crop word images
    word_images = []
    for x1, x2 in words:
        word = line_img[:, x1:x2]
        word_images.append(word)

    return word_images, words
def cc_word_segmentation(bin_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)

    words = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area > 50:  # filter noise
            words.append(bin_img[y:y+h, x:x+w])

    return words

word_images, words = word_segmentation(cv2.imread('images/testml5.jpg'))
print(f"Detected {len(word_images)} words at positions: {words}")


from utils import imagePlot
imagePlot.plot_images(word_images, titles=[f"Word {i+1}" for i in range(len(word_images))])