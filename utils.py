from matplotlib import pyplot as plt
import numpy as np
import pytesseract
import cv2


class Utils:
    @staticmethod
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def threshold(image, max_value=255, block_size=11, C=2):
        return cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    
    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path)
    
    @staticmethod
    def ocr(image):
        return pytesseract.image_to_string(image, lang='mal', config="--oem 1 --psm 7"
)

    @staticmethod
    def deionise(image,h,templateWindowSize,searchWindowSize):
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h,     # h strength
            templateWindowSize,      # templateWindowSize
            searchWindowSize      # searchWindowSize
        )
    @staticmethod
    def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    @staticmethod   
    def morphological_operations(image, kernel_size=(3, 3), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    @staticmethod 
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

class imagePlot:
    @staticmethod
    def plot_images(images, titles=None):
        n = len(images)
        plt.figure(figsize=(15, 5))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(images[i], cmap='gray')
            if titles:
                plt.title(titles[i])
            plt.axis('off')
        plt.show()
    
    def show_image(image, title=None):
        plt.imshow(image, cmap='gray')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

class WordCorpus:
    
    @staticmethod
    def histogram(image):
        # Horizontal projection
        _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
        return np.sum(binary == 0, axis=1)

    @staticmethod
    def crop_text(image, hist):
        lines = []

        in_line = False
        start = 0
        threshold = 8

        for i, val in enumerate(hist):

            if val > threshold and not in_line:
                in_line = True
                start = i

            elif val == threshold and in_line:
                in_line = False
                end = i
                lines.append(image[start:end, :])

        # Handle last line if image ends before returning to 0
        if in_line:
            lines.append(image[start:, :])

        return lines