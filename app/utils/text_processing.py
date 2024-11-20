import re
from googletrans import Translator

def preprocess_text(text: str):
        translator = Translator()
        translated = translator.translate(text, dest='en').text
        return translated


def extract_words(text: str) -> set:
    """Extract words from text"""
    return set(re.findall(r'\b\w+\b', text.lower()))

ans = preprocess_text('সকাল থেকে পেটে ব্যথা করতেছে, বিকেলে মাথাব্যথা করছিল, রাতে বমি হয়েছে ।')
print(ans)




