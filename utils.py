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

    @staticmethod
    def sort_bounding_boxes_2d(results_list):
        """
        Sorts PaddleOCR boxes vertically (line-by-line) and horizontally (word-by-word).
        Handles uneven text lines caused by shadows, tilt, or varying font sizes.
        """
        if not results_list:
            return []

        # 1. Map results to calculate bounding properties: (y_center, x_min, original_item)
        box_data = []
        for res in results_list:
            box = res[0]
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            # Using the vertical center makes row grouping much more stable than y_min
            y_center = (y_min + y_max) / 2
            
            box_data.append({
                'y_center': y_center,
                'x_min': x_min,
                'height': y_max - y_min,
                'raw': res
            })

        # 2. Sort primarily by vertical center to start grouping rows top-to-bottom
        box_data.sort(key=lambda item: item['y_center'])

        rows = []
        for box in box_data:
            placed = False
            # Check if this box fits into an existing row based on vertical overlap
            for row in rows:
                # We calculate the average height of the current row to make the threshold dynamic
                avg_row_height = sum(b['height'] for b in row) / len(row)
                # Dynamic threshold: allow a variance of 60% of the average text height
                row_threshold = avg_row_height * 0.6 
                
                # If the box's center is vertically aligned with this row, add it
                row_center = sum(b['y_center'] for b in row) / len(row)
                if abs(box['y_center'] - row_center) < row_threshold:
                    row.append(box)
                    placed = True
                    break
            
            # If it doesn't fit any existing row, create a new row
            if not placed:
                rows.append([box])

        # 3. Sort the rows themselves top-to-bottom
        rows.sort(key=lambda r: sum(b['y_center'] for b in r) / len(r))

        # 4. For each row, sort the words horizontally from left to right
        sorted_results = []
        for row in rows:
            row.sort(key=lambda b: b['x_min'])
            sorted_results.extend([b['raw'] for b in row])

        return sorted_results
    
    @staticmethod
    def reconstruct_lines(texts_with_positions):
        if not texts_with_positions:
            return []

        # Calculate average box height to dynamically set the Y-threshold
        heights = []
        for box, _ in texts_with_positions:
            y_coords = [p[1] for p in box]
            heights.append(max(y_coords) - min(y_coords))
        avg_height = sum(heights) / len(heights) if heights else 20
        y_threshold = avg_height * 0.5  # 50% of the average line height is highly robust

        # Sort primarily from top to bottom based on the top-Y coordinate
        sorted_by_y = sorted(texts_with_positions, key=lambda x: min(p[1] for p in x[0]))

        lines = []
        current_line = []
        current_y_center = None

        for box, text in sorted_by_y:
            y_coords = [p[1] for p in box]
            y_min, y_max = min(y_coords), max(y_coords)
            y_center = (y_min + y_max) / 2

            if current_y_center is None:
                current_y_center = y_center
                current_line.append((box, text))
            elif abs(y_center - current_y_center) < y_threshold:
                # Belongs to the same horizontal line
                current_line.append((box, text))
            else:
                # End of a line reached. Sort the line's words left-to-right (X-coordinate)
                current_line_sorted = sorted(current_line, key=lambda x: min(p[0] for p in x[0]))
                line_text = " ".join([item[1] for item in current_line_sorted])
                lines.append(line_text)

                # Start a new line
                current_y_center = y_center
                current_line = [(box, text)]

        # Append the final line
        if current_line:
            current_line_sorted = sorted(current_line, key=lambda x: min(p[0] for p in x[0]))
            line_text = " ".join([item[1] for item in current_line_sorted])
            lines.append(line_text)

        return lines
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
        threshold = 0

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