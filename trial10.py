# import cv2
# import re
# import os
# import pytesseract
# import concurrent.futures
# from paddleocr import PaddleOCR
# from utils import imagePlot, Utils

# # ==========================================
# # 1. MALAYALAM AUTO-CORRECT DICTIONARY
# # ==========================================
# MALAYALAM_AUTO_CORRECT = {
#     "വഞ്ചായത്ത്": "പഞ്ചായത്ത്",
#     "വഞ്ചായത്തുകള്": "പഞ്ചായത്തുകൾ",
#     "വഞ്ചായത്തുകളുടെയും": "പഞ്ചായത്തുകളുടെയും",
#     "വകര്പ്പ്": "പകർപ്പ്",
#     "വൂര്ത്തി": "പൂർത്തി",
#     "വങ്കെടു": "പങ്കെടു",
#     "വിഫിതം": "വിഹിതം",
#     "ധനസഫായം": "ധനസഹായം",
#     "തീഡിയ": "മീഡിയ"
# }
# import requests

# def autocorrect_malayalam_api(text: str) -> str:
#     """Queries the LanguageTool API to spellcheck and correct Malayalam text."""
#     url = "https://api.languagetool.org/v2/check"
#     data = {
#         'text': text,
#         'language': 'ml'
#     }
    
#     try:
#         response = requests.post(url, data=data, timeout=5)
#         if response.status_code == 200:
#             result = response.json()
#             matches = result.get('matches', [])
            
#             # Apply corrections in reverse order to preserve string offsets
#             text_chars = list(text)
#             for match in sorted(matches, key=lambda x: x['offset'], reverse=True):
#                 offset = match['offset']
#                 length = match['length']
#                 replacements = match['replacements']
                
#                 if replacements:
#                     best_replacement = replacements[0]['value']
#                     text_chars[offset : offset + length] = list(best_replacement)
            
#             return "".join(text_chars)
#     except Exception as e:
#         print(f"[API Error] Spellcheck failed, falling back to raw text: {e}")
        
#     return text
# def clean_malayalam_errors(text: str) -> str:
#     """Corrects predictable Malayalam character substitutions."""
#     for error_word, correct_word in MALAYALAM_AUTO_CORRECT.items():
#         text = re.sub(rf'\b{error_word}', correct_word, text)
#     return text

# def reconstruct_lines_local(texts_with_positions):
#     """Groups word segments back into horizontal lines based on their vertical alignment."""
#     if not texts_with_positions:
#         return []

#     # Calculate average box height to dynamically set the Y-threshold
#     heights = []
#     for box, _ in texts_with_positions:
#         y_coords = [p[1] for p in box]
#         heights.append(max(y_coords) - min(y_coords))
#     avg_height = sum(heights) / len(heights) if heights else 20
#     y_threshold = avg_height * 0.5  

#     # Sort primarily from top to bottom
#     sorted_by_y = sorted(texts_with_positions, key=lambda x: min(p[1] for p in x[0]))

#     lines = []
#     current_line = []
#     current_y_center = None

#     for box, text in sorted_by_y:
#         y_coords = [p[1] for p in box]
#         y_min, y_max = min(y_coords), max(y_coords)
#         y_center = (y_min + y_max) / 2

#         if current_y_center is None:
#             current_y_center = y_center
#             current_line.append((box, text))
#         elif abs(y_center - current_y_center) < y_threshold:
#             current_line.append((box, text))
#         else:
#             current_line_sorted = sorted(current_line, key=lambda x: min(p[0] for p in x[0]))
#             line_text = " ".join([item[1] for item in current_line_sorted])
#             lines.append(line_text)

#             current_y_center = y_center
#             current_line = [(box, text)]

#     if current_line:
#         current_line_sorted = sorted(current_line, key=lambda x: min(p[0] for p in x[0]))
#         line_text = " ".join([item[1] for item in current_line_sorted])
#         lines.append(line_text)

#     return lines

# def ocr_single_crop(item):
#     """Worker function to process and OCR a single crop."""
#     box, cropped, img_w, img_h = item
#     try:
#         # --- Preprocessing ---
#         gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
#         # Add a generous white border to assist Tesseract's page layout analysis (PSM 7)
#         gray_crop = cv2.copyMakeBorder(gray_crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255])
        
#         # Conditional upscaling (only upscale if the crop is small to save CPU time)
#         h, w = gray_crop.shape[:2]
#         if h < 40:
#             gray_crop = cv2.resize(gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
#         blurred = cv2.GaussianBlur(gray_crop, (3, 3), 0)
        
#         # Use Otsu's Binarization for cleaner, solid text lines
#         _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # Ensure text is black on a white background
#         if cv2.mean(threshold)[0] < 127:
#             threshold = cv2.bitwise_not(threshold)
            
#         # Run Tesseract with Single-Line mode (PSM 7)
#         custom_config = r'--oem 1 --psm 7 --tessdata-dir "custom_tessdata"'
#         text = pytesseract.image_to_string(threshold, lang='mal', config=custom_config)
        
#         # Clean up whitespace and line breaks
#         text = text.replace('\x0c', '').replace('\n', ' ')
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         return box, text
#     except Exception as e:
#         print(f"[Thread Error] Failed on a crop: {str(e)}")
#         return box, ""


# # ==========================================
# # 2. MAIN EXECUTION PIPELINE
# # ==========================================

# print("[1/6] Initializing PaddleOCR...")
# ocr = PaddleOCR(
#     use_angle_cls=True, 
#     lang='ta',
#     det_db_box_thresh=0.5,   
#     det_db_unclip_ratio=1.6  
# )

# # Set your target image path
# image_path = 'images/Ups.jpeg'  

# if not os.path.exists(image_path):
#     print(f"[Error] Image file not found at: {image_path}")
#     exit()

# print(f"[2/6] Loading image: {image_path}...")
# original_image = cv2.imread(image_path)
# img_h, img_w = original_image.shape[:2]

# # ==========================================
# # PRE-PROCESSING STEP FOR PADDLEOCR DETECTION
# # ==========================================
# print("[3/6] Preprocessing main image...")
# gray_main = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
# background = cv2.morphologyEx(gray_main, cv2.MORPH_DILATE, kernel_bg)
# background = cv2.GaussianBlur(background, (51, 51), 0)
# normalized_gray = cv2.divide(gray_main, background, scale=255)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced_gray = clahe.apply(normalized_gray)
# detection_ready_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# # ==========================================
# # RUN PADDLEOCR DETECTION
# # ==========================================
# print("[4/6] Running PaddleOCR text detector...")
# dt_boxes, elapse = ocr.text_detector(detection_ready_image)

# if dt_boxes is None or len(dt_boxes) == 0:
#     print("[Exit] No text boxes detected by PaddleOCR.")
#     exit()

# print(f"-> Detected {len(dt_boxes)} raw text boundaries.")

# formatted_results = [[box.tolist(), ("", 0.0)] for box in dt_boxes]
# sorted_line_blocks = Utils.sort_bounding_boxes_2d(formatted_results)

# # ==========================================
# # PREPARE IMAGE CROPS QUEUE
# # ==========================================
# crops_queue = []
# for result in sorted_line_blocks:
#     box = result[0]    
#     x_coords = [point[0] for point in box]
#     y_coords = [point[1] for point in box]
    
#     padding_x = 8
#     padding_y = 6
    
#     x_min = max(0, int(min(x_coords)) - padding_x)
#     x_max = min(img_w, int(max(x_coords)) + padding_x)
#     y_min = max(0, int(min(y_coords)) - padding_y)
#     y_max = min(img_h, int(max(y_coords)) + padding_y)
    
#     cropped = original_image[y_min:y_max, x_min:x_max]
    
#     if cropped.size > 0:
#         crops_queue.append((box, cropped, img_w, img_h))

# print(f"-> Queued {len(crops_queue)} valid image crops for Tesseract.")

# # ==========================================
# # RUN PARALLEL TESSERACT EXTRACTION (Safe & Logged)
# # ==========================================
# print("[5/6] Starting Tesseract extraction in parallel threads...")
# texts_with_positions = []
# total_crops = len(crops_queue)
# completed_count = 0

# # Limit max_workers to 4 to prevent process deadlocks and CPU thrashing
# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     # Submit tasks
#     future_to_crop = {executor.submit(ocr_single_crop, crop): crop for crop in crops_queue}
    
#     # Process results in real-time as they complete
#     for future in concurrent.futures.as_completed(future_to_crop):
#         try:
#             box, text = future.result()
#             if text:
#                 texts_with_positions.append((box, text))
#         except Exception as exc:
#             print(f"[Thread Error] Generated an exception: {exc}")
            
#         completed_count += 1
#         # Print progress every 10 crops
#         if completed_count % 10 == 0 or completed_count == total_crops:
#             print(f"   -> Progress: Processed {completed_count}/{total_crops} crops...")

# print(f"-> Successfully extracted text from {len(texts_with_positions)} crops.")

# # ==========================================
# # 6. LINE RECONSTRUCTION & FILE SAVE
# # ==========================================
# print("[6/6] Reconstructing and saving output layout...")

# # Group isolated words back into structured lines
# structured_lines = reconstruct_lines_local(texts_with_positions)

# if not structured_lines:
#     print("[Exit] Line reconstruction returned 0 lines.")
#     exit()

# # Apply standard dictionary corrections (e.g., വഞ്ചായത്ത് -> പഞ്ചായത്ത്)
# cleaned_lines = [autocorrect_malayalam_api(line) for line in structured_lines]

# # === THE FILE SAVE PART ===
# output_file = 'extracted_texts_new.txt'
# with open(output_file, 'w', encoding='utf-8') as f:
#     for line in cleaned_lines:
#         f.write(line + '\n')

# print(f"\n[Success] Process complete! Cleaned text saved to '{output_file}'.")


import cv2
import re
import os
import requests
import pytesseract
import concurrent.futures
from paddleocr import PaddleOCR
from utils import imagePlot, Utils

# ==========================================
# 1. AUTOCORRECT SYSTEM (LOCAL & API)
# ==========================================

# High-speed local corrections used as a zero-latency fallback if the API times out
MALAYALAM_AUTO_CORRECT = {
    "വഞ്ചായത്ത്": "പഞ്ചായത്ത്",
    "വഞ്ചായത്തുകള്": "പഞ്ചായത്തുകൾ",
    "വഞ്ചായത്തുകളുടെയും": "പഞ്ചായത്തുകളുടെയും",
    "വകര്പ്പ്": "പകർപ്പ്",
    "വൂര്ത്തി": "പൂർത്തി",
    "വങ്കെടു": "പങ്കെടു",
    "വിഫിതം": "വിഹിതം",
    "ധനസഫായം": "ധനസഹായം",
    "തീഡിയ": "മീഡിയ"
}

def clean_malayalam_errors_local(text: str) -> str:
    """Corrects predictable Malayalam character substitutions using reliable direct string replacement."""
    for error_word, correct_word in MALAYALAM_AUTO_CORRECT.items():
        text = text.replace(error_word, correct_word)
    return text

def autocorrect_malayalam_api(text: str) -> str:
    """Queries the LanguageTool API to spellcheck and correct the entire document."""
    url = "https://api.languagetool.org/v2/check"
    data = {
        'text': text,
        'language': 'ml'
    }
    
    try:
        response = requests.post(url, data=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            matches = result.get('matches', [])
            
            # Apply corrections in reverse order to preserve string offsets
            text_chars = list(text)
            for match in sorted(matches, key=lambda x: x['offset'], reverse=True):
                offset = match['offset']
                length = match['length']
                replacements = match['replacements']
                
                if replacements:
                    best_replacement = replacements[0]['value']
                    text_chars[offset : offset + length] = list(best_replacement)
            
            return "".join(text_chars)
    except Exception as e:
        print(f"[API Warning] Spellcheck API failed: {e}. Falling back to local dictionary corrections.")
        
    # Fallback to local dictionary if the network request fails or times out
    return clean_malayalam_errors_local(text)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def reconstruct_lines_local(texts_with_positions):
    """Groups word segments back into horizontal lines based on their vertical alignment."""
    if not texts_with_positions:
        return []

    # Calculate average box height to dynamically set the Y-threshold
    heights = []
    for box, _ in texts_with_positions:
        y_coords = [p[1] for p in box]
        heights.append(max(y_coords) - min(y_coords))
    avg_height = sum(heights) / len(heights) if heights else 20
    y_threshold = avg_height * 0.5  

    # Sort primarily from top to bottom
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
            current_line.append((box, text))
        else:
            current_line_sorted = sorted(current_line, key=lambda x: min(p[0] for p in x[0]))
            line_text = " ".join([item[1] for item in current_line_sorted])
            lines.append(line_text)

            current_y_center = y_center
            current_line = [(box, text)]

    if current_line:
        current_line_sorted = sorted(current_line, key=lambda x: min(p[0] for p in x[0]))
        line_text = " ".join([item[1] for item in current_line_sorted])
        lines.append(line_text)

    return lines

def ocr_single_crop(item):
    """Worker function to process and OCR a single crop in parallel."""
    box, cropped, img_w, img_h = item
    try:
        # --- Preprocessing ---
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Add a generous white border to assist Tesseract's page layout analysis (PSM 7)
        gray_crop = cv2.copyMakeBorder(gray_crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255])
        
        # Conditional upscaling (only upscale if the crop is small to save CPU time)
        h, w = gray_crop.shape[:2]
        if h < 40:
            gray_crop = cv2.resize(gray_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
        blurred = cv2.GaussianBlur(gray_crop, (3, 3), 0)
        
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
        
        return box, text
    except Exception as e:
        print(f"[Thread Error] Failed on a crop: {str(e)}")
        return box, ""


# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

print("[1/6] Initializing PaddleOCR...")
# Changed lang to 'ml' (Malayalam) to align with document script
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='ta',
    det_db_box_thresh=0.5,   
    det_db_unclip_ratio=1.6  
)

# Set your target image path
image_path = 'images/Screenshot_2026-02-02-15-19-19-31_f73b71075b1de7323614b647fe394240.jpg'  

if not os.path.exists(image_path):
    print(f"[Error] Image file not found at: {image_path}")
    exit()

print(f"[2/6] Loading image: {image_path}...")
original_image = cv2.imread(image_path)
img_h, img_w = original_image.shape[:2]

# ==========================================
# PRE-PROCESSING STEP FOR PADDLEOCR DETECTION
# ==========================================
print("[3/6] Preprocessing main image...")
gray_main = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
background = cv2.morphologyEx(gray_main, cv2.MORPH_DILATE, kernel_bg)
background = cv2.GaussianBlur(background, (51, 51), 0)
normalized_gray = cv2.divide(gray_main, background, scale=255)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(normalized_gray)
detection_ready_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# ==========================================
# RUN PADDLEOCR DETECTION
# ==========================================
print("[4/6] Running PaddleOCR text detector...")
dt_boxes, elapse = ocr.text_detector(detection_ready_image)

if dt_boxes is None or len(dt_boxes) == 0:
    print("[Exit] No text boxes detected by PaddleOCR.")
    exit()

print(f"-> Detected {len(dt_boxes)} raw text boundaries.")

formatted_results = [[box.tolist(), ("", 0.0)] for box in dt_boxes]
sorted_line_blocks = Utils.sort_bounding_boxes_2d(formatted_results)

# ==========================================
# PREPARE IMAGE CROPS QUEUE
# ==========================================
crops_queue = []
for result in sorted_line_blocks:
    box = result[0]    
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    
    padding_x = 8
    padding_y = 6
    
    x_min = max(0, int(min(x_coords)) - padding_x)
    x_max = min(img_w, int(max(x_coords)) + padding_x)
    y_min = max(0, int(min(y_coords)) - padding_y)
    y_max = min(img_h, int(max(y_coords)) + padding_y)
    
    cropped = original_image[y_min:y_max, x_min:x_max]
    
    if cropped.size > 0:
        crops_queue.append((box, cropped, img_w, img_h))

print(f"-> Queued {len(crops_queue)} valid image crops for Tesseract.")

# ==========================================
# RUN PARALLEL TESSERACT EXTRACTION (Safe & Logged)
# ==========================================
print("[5/6] Starting Tesseract extraction in parallel threads...")
texts_with_positions = []
total_crops = len(crops_queue)
completed_count = 0

# Limit max_workers to 4 to prevent process deadlocks and CPU thrashing
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    future_to_crop = {executor.submit(ocr_single_crop, crop): crop for crop in crops_queue}
    
    # Process results in real-time as they complete
    for future in concurrent.futures.as_completed(future_to_crop):
        try:
            box, text = future.result()
            if text:
                texts_with_positions.append((box, text))
        except Exception as exc:
            print(f"[Thread Error] Generated an exception: {exc}")
            
        completed_count += 1
        # Print progress every 10 crops
        if completed_count % 10 == 0 or completed_count == total_crops:
            print(f"   -> Progress: Processed {completed_count}/{total_crops} crops...")

print(f"-> Successfully extracted text from {len(texts_with_positions)} crops.")

# ==========================================
# 6. LINE RECONSTRUCTION, API AUTOCORRECT & FILE SAVE
# ==========================================
print("[6/6] Reconstructing and saving output layout...")

# Group isolated words back into structured lines
structured_lines = reconstruct_lines_local(texts_with_positions)

if not structured_lines:
    print("[Exit] Line reconstruction returned 0 lines.")
    exit()

print("-> Running dynamic autocorrect on the entire document...")

# Merging all lines with newline delimiters to process everything in exactly ONE API call
document_text = "\n".join(structured_lines)
cleaned_document = autocorrect_malayalam_api(document_text)

# Split the cleaned document back into individual lines
cleaned_lines = cleaned_document.split("\n")

# === THE FILE SAVE PART ===
output_file = 'extracted_texts_new.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print(f"\n[Success] Process complete! Cleaned text saved to '{output_file}'.")