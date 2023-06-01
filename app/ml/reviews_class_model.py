import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from app.ml.text_classifier_model import TextClassifier


class ReviewsClassModel(TextClassifier):
    def create_model(self, df: pd.DataFrame):
        df = self.preprocess(df)
        # df = df.drop(df[df["class"] == 0].sample(frac=0.5).index)

        X_train, X_test, y_train, y_test = self.prepare_data(df, "class", 40, 0.7)

        model = Sequential()
        model.add(Embedding(len(self.word_2_index), 50))
        model.add(SpatialDropout1D(0.7))
        model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
        model.add(Dense(3, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        model.fit(
            X_train,
            y_train,
            batch_size=8,
            epochs=20,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        model.save(self.model_path)

