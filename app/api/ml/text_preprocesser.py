import nltk
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = set(nltk.corpus.words.words())
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text):
        text = re.sub("[\(\[].*?[\)\]]", "", str(text))
        text = re.sub(r"[^\w\s]", "", str(text))
        text = text.lower()
        text = [
            self.lemmatizer.lemmatize(word)
            for word in word_tokenize(text)
            if self.lemmatizer.lemmatize(word) not in self.stop_words
            and word in self.words
        ]
        text = " ".join(text)

        return text
