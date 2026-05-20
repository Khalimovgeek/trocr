import requests

def get_corrected_text(ocr_text):
    url = "https://api.languagetoolplus.com/v2/check"
    data = {
        'text': ocr_text,
        'language': 'ml',
        'enabledOnly': 'false'
    }
    
    response = requests.post(url, data=data)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: Server returned status {response.status_code}")
        print(f"Response Body: {response.text}") # This will reveal if it's a 403, 429, or 500 error
        return ocr_text # Return original text on failure
    
    result = response.json()
    # ... rest of your code

# Your OCR text
raw_ocr = ""
with open('output.txt', 'r', encoding='utf-8') as f:
    raw_ocr = f.read()

corrected = get_corrected_text(raw_ocr)
print(f"Original: {raw_ocr}")
print(f"Corrected: {corrected}")