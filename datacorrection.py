# import requests

# def get_corrected_text(ocr_text):
#     url = "https://api.languagetoolplus.com/v2/check"
#     data = {
#         'text': ocr_text,
#         'language': 'ml',
#         'enabledOnly': 'false'
#     }
    
#     response = requests.post(url, data=data)
    
#     # Check if the request was successful
#     if response.status_code != 200:
#         print(f"Error: Server returned status {response.status_code}")
#         print(f"Response Body: {response.text}") # This will reveal if it's a 403, 429, or 500 error
#         return ocr_text # Return original text on failure
    
#     result = response.json()
#     # ... rest of your code

# # Your OCR text
# raw_ocr = ""
# with open('output.txt', 'r', encoding='utf-8') as f:
#     raw_ocr = f.read()

# corrected = get_corrected_text(raw_ocr)
# print(f"Original: {raw_ocr}")
# print(f"Corrected: {corrected}")

import hunspell

# Initialize Hunspell with Malayalam dictionary files
# Ubuntu installs these to /usr/share/hunspell/
dic_path = '/usr/share/hunspell/ml_IN.dic'
aff_path = '/usr/share/hunspell/ml_IN.aff'

checker = hunspell.HunSpell(dic_path, aff_path)

def correct_text(text):
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Check if the word is spelled correctly
        if checker.spell(word):
            corrected_words.append(word)
        else:
            # Get suggestions
            suggestions = checker.suggest(word)
            if suggestions:
                # Use the first suggestion
                corrected_words.append(suggestions[0])
            else:
                corrected_words.append(word)
                
    return " ".join(corrected_words)

# Example usage in your script:
# original_text = "ഇടിച്ചു (ഓടനാടക്റേ രാജവംശം] ..."
# corrected_text = correct_text(original_text)
# print("Original:", original_text)
# print("Corrected:", corrected_text)

# # Your OCR text
raw_ocr = ""
with open('output.txt', 'r', encoding='utf-8') as f:
    raw_ocr = f.read()

corrected = correct_text(raw_ocr)
print(f"Original: {raw_ocr}")
print(f"Corrected: {corrected}")