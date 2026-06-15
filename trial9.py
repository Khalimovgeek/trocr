# from paddleocr import PaddleOCR
# import cv2
# import re
# import pytesseract
# from utils import imagePlot, Utils

# # 1. Keep PaddleOCR initialization clean and close to default
# # Let it focus on structure now that the image will be clean
# ocr = PaddleOCR(
#     use_angle_cls=True, 
#     lang='ta',
#     #det_db_thresh=0.3,       # Reset to stable defaults
#     det_db_box_thresh=0.5,   
#     det_db_unclip_ratio=1.6  # Standard unclip works well on uniform backgrounds
# )

# image_path = 'images/page_2.jpg'
# original_image = cv2.imread(image_path)

# # ==========================================
# # PRE-PROCESSING STEP FOR PADDLEOCR DETECTION
# # ==========================================
# # A. Convert to grayscale
# gray_main = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# # B. Remove uneven lighting/shadows using morphological opening
# # We use a large kernel to capture the general background illumination gradient
# kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
# background = cv2.morphologyEx(gray_main, cv2.MORPH_DILATE, kernel_bg)
# background = cv2.GaussianBlur(background, (51, 51), 0)

# # C. Subtract background to achieve uniform lightning across the page
# normalized_gray = cv2.divide(gray_main, background, scale=255)

# # D. Denoise and sharpen text contrast using CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced_gray = clahe.apply(normalized_gray)

# # E. Convert back to 3-channel BGR (PaddleOCR expects BGR format)
# detection_ready_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# # Optional: Inspect this image to see how pristine the text blocks became!
# # imagePlot.show_image(detection_ready_image, title="Enhanced for PaddleOCR")

# # ==========================================
# # RUN PADDLEOCR ON THE ENHANCED IMAGE
# # ==========================================
# results = ocr.ocr(detection_ready_image)

# texts = []
# img_h, img_w = original_image.shape[:2]

# # (Make sure to use your 2D sorting function here)
# sorted_line_blocks = Utils.sort_bounding_boxes_2d(results[0])

# for result in sorted_line_blocks:
#     box = result[0]
    
#     x_coords = [point[0] for point in box]
#     y_coords = [point[1] for point in box]
    
#     # Crop out of the ORIGINAL image (or enhanced_gray depending on what your Tesseract prefers)
#     padding = 6
#     x_min = max(0, int(min(x_coords)) - padding)
#     x_max = min(img_w, int(max(x_coords)) + padding)
#     y_min = max(0, int(min(y_coords)) - padding)
#     y_max = min(img_h, int(max(y_coords)) + padding)
    
#     cropped = original_image[y_min:y_max, x_min:x_max]
#     cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     cv2.imwrite('bounding box.jpg', original_image)
#     # ==========================================
#     # TESSERACT LINE PREPROCESSING
#     # ==========================================
#     gray_crop = Utils.gray_scale(cropped)
#     gray_crop = cv2.copyMakeBorder(gray_crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#     upscaled = cv2.resize(gray_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#     blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
    
#     threshold = Utils.threshold(blurred, max_value=255, block_size=15, C=3)
#     if cv2.mean(threshold)[0] < 127:
#         threshold = cv2.bitwise_not(threshold)
        
#     custom_config = r'--oem 1 --psm 7 --tessdata-dir "custom_tessdata"'
#     text = pytesseract.image_to_string(threshold, lang='mal', config=custom_config)
    
#     text = text.replace('\x0c', '').replace('\n', ' ')
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     if text:
#         texts.append(text)

# print(texts)

import cv2
import re
import pytesseract
from paddleocr import PaddleOCR
from utils import imagePlot, Utils

# 1. Initialize PaddleOCR for detection only
# Setting lang='ml' (Malayalam) or 'en' is standard. 
# We will use rec=False in the detection call to bypass the recognition step.
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='ta',
    det_db_box_thresh=0.5,   
    det_db_unclip_ratio=1.6  
)

image_path = 'images/page_1.jpg'
original_image = cv2.imread(image_path)
img_h, img_w = original_image.shape[:2]

# ==========================================
# PRE-PROCESSING STEP FOR PADDLEOCR DETECTION
# ==========================================
gray_main = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Check if image is already high-contrast/clean to avoid over-processing
# If the image has shadows, keep this background normalization active.
kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
background = cv2.morphologyEx(gray_main, cv2.MORPH_DILATE, kernel_bg)
background = cv2.GaussianBlur(background, (51, 51), 0)
normalized_gray = cv2.divide(gray_main, background, scale=255)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(normalized_gray)

# PaddleOCR expects a 3-channel image
detection_ready_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# ==========================================
# RUN PADDLEOCR DETECTION (rec=False for speed)
# ==========================================
# Setting rec=False speeds up processing significantly
# ==========================================
# RUN PADDLEOCR DETECTION (Direct & Bug-Free)
# ==========================================
# Calling the internal text_detector directly bypasses the high-level NumPy bug
dt_boxes, elapse = ocr.text_detector(detection_ready_image)

if dt_boxes is None or len(dt_boxes) == 0:
    print("No text blocks detected.")
    exit()

# Convert the NumPy arrays to lists (.tolist()) to prevent similar NumPy 
# evaluation errors in downstream custom sorting utility functions.
formatted_results = [[box.tolist(), ("", 0.0)] for box in dt_boxes]

# Sort the bounding boxes using your 2D sorting function
sorted_line_blocks = Utils.sort_bounding_boxes_2d(formatted_results)
# texts = []
texts_with_positions = []
for result in sorted_line_blocks:
    box = result[0]
    
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    
    # Crop padding - slightly increased to avoid clipping character ascenders/descenders
    padding_x = 8
    padding_y = 6
    
    x_min = max(0, int(min(x_coords)) - padding_x)
    x_max = min(img_w, int(max(x_coords)) + padding_x)
    y_min = max(0, int(min(y_coords)) - padding_y)
    y_max = min(img_h, int(max(y_coords)) + padding_y)
    
    cropped = original_image[y_min:y_max, x_min:x_max]
    
    # Optional: Draw bounding box for debugging
    # cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # ==========================================
    # TESSERACT LINE PREPROCESSING
    # ==========================================
    if cropped.size == 0:
        continue
        
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Add a generous white border to assist Tesseract's page layout analysis (PSM 7)
    gray_crop = cv2.copyMakeBorder(gray_crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255])
    
    # Upscale to make small text easier to read
    upscaled = cv2.resize(gray_crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
    
    # Use Otsu's Binarization for cleaner, solid text lines
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ensure text is black on a white background
    if cv2.mean(threshold)[0] < 127:
        threshold = cv2.bitwise_not(threshold)
        
    # Run Tesseract with Single-Line mode (PSM 7)
    custom_config = r'--oem 1 --psm 7 --tessdata-dir "custom_tessdata"'
    text = pytesseract.image_to_string(threshold, lang='mal', config=custom_config)
    
    # Clean up whitespace and line breaks
    text = text.replace('\x0c', '').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text:
        texts_with_positions.append((box, text))


# # Save the bounding box image to verify detection coverage
# cv2.imwrite('bounding_box_output.jpg', original_image)

# Reconstruct the text into proper lines
structured_lines = Utils.reconstruct_lines(texts_with_positions)

# Write to the file
with open('extracted_texts.txt', 'w', encoding='utf-8') as f:
    for line in structured_lines:
        f.write(line + '\n')

