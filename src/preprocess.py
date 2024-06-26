# preprocess.py

import regex as re
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

class Preprocess:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]', '', text)
        # Convert to lowercase (if not Japanese)
        text = text.lower()
        # Tokenize text
        words = text.split()
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)

    def apply_preprocessing(self, data):
        data['clean_subject'] = data['subject'].apply(self.preprocess_text)
        data['length'] = data['subject'].apply(len)
        data['digit_count'] = data['subject'].apply(lambda x: len(re.findall(r'\d', x)))
        data['exclamation_count'] = data['subject'].apply(lambda x: x.count('！'))
        data['tone'] = data['tone']
        data['style'] = data['style']
        return data
