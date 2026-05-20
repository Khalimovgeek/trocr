from rapidfuzz import process, utils

# The vocabulary from your Ground Truth
vocabulary = [
    "വീണ്ടും", "കാട്ടാന", "വൈദ്യുതി", "വേലി", "തകർത്തു", "കാട്ടിലും", 
    "നാട്ടിലും", "നാശം", "വിതച്ചു", "ആയമ്പുഴ", "ജംങ്ഷനു", "സമീപമുള്ള", 
    "പത്തോളം", "വീടുകളിലെ", "വാഴകളും", "അടയ്ക്കാമരങ്ങളും", "മറ്റും", 
    "പ്ലാന്റേഷൻ", "കോർപ്പറേഷന്റെ", "എണ്ണപ്പന", "തോട്ടത്തിൽ", "നിന്നാണ്"
]

def clean_ocr_output(garbled_word):
    # Find the closest match in the ground truth vocabulary
    # score_cutoff=50 prevents it from 'hallucinating' a match for total noise
    match = process.extractOne(garbled_word, vocabulary, processor=utils.default_process, score_cutoff=70)
    
    if match:
        return match[0]
    return garbled_word # Keep original if no close match found

# Example usage:
# If Tesseract gives 'ചീണ്ദും', clean_ocr_output('ചീണ്ദും') -> 'വീണ്ടും'


with open('output.txt', 'r', encoding='utf-8') as f:
    ocr_output = f.read()
    ocr_clean  = []
    for word in ocr_output.split():
        cleaned_word = clean_ocr_output(word)
        print(f"Original: {word} -> Cleaned: {cleaned_word}")
        ocr_clean.append(cleaned_word)
    with open('cleaned_output.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(ocr_clean))    