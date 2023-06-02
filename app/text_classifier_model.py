import nltk
import pandas as pd
import numpy as np
import keras
from keras.utils import pad_sequences
from nltk import word_tokenize
from sklearn.model_selection import train_test_split

from text_preprocessor import TextPreprocessor
import os

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("words", quiet=True)


class TextClassifier:
    word_2_index = {}
    index_2_class = []
    model_path = ""
    text_preprocessor = TextPreprocessor()
    padding_maxlen = 100

    def __init__(self, model_path):
        self.model_path = model_path

    def preprocess(self, df: pd.DataFrame):
        df["text"] = df["text"].apply(self.text_preprocessor.preprocess)

        text_words = word_tokenize(" ".join(df["text"].to_numpy()))
        unique_words = set(text_words)

        self.word_2_index = {w: i for i, w in enumerate(unique_words)}

        df["embedding"] = [
            [self.word_2_index[word] for word in i] for i in df["text"].str.split()
        ]
        df = df[df["embedding"].notna()]

        return df

    def prepare_data(
        self, df: pd.DataFrame, target: str, padding_maxlen: int, train_size: float
    ):
        self.padding_maxlen = padding_maxlen
        targets = pd.get_dummies(df[target])
        self.index_2_class = targets.columns.astype(str)
        X, y = df["embedding"], targets.values
        X = pad_sequences(X, maxlen=self.padding_maxlen)

        return train_test_split(X, y, train_size=train_size)

    def predict(self, df: pd.DataFrame, prediction_column: str):
        model = keras.models.load_model(os.getcwd() + '/' + self.model_path)
        df["text"] = df["text"].apply(self.text_preprocessor.preprocess)
        df["embedding"] = [
            [self.word_2_index[word] if word in self.word_2_index else 0 for word in i]
            for i in df["text"].str.split()
        ]
        df = df[df["embedding"].notna()]
        embeddings = pad_sequences(df["embedding"], maxlen=self.padding_maxlen)
        predictions = model.predict(embeddings)
        df["predictions"] = [p for p in predictions]
        df[prediction_column] = [
            self.index_2_class[np.argmax(prediction)] for prediction in predictions
        ]

        return df