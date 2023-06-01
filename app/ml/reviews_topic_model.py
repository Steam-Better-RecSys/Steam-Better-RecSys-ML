import joblib
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from app.ml.text_classifier_model import TextClassifier


class ReviewsTopicModel(TextClassifier):
    def create_model(self, df: pd.DataFrame):
        df = df[df["topic"].notna()]
        df = self.preprocess(df)

        X_train, X_test, y_train, y_test = self.prepare_data(df, "topic", 30, 0.8)

        model = Sequential()
        model.add(Embedding(len(self.word_2_index), 100))
        model.add(SpatialDropout1D(0.8))
        model.add(LSTM(64, dropout=0.8, recurrent_dropout=0.8))
        model.add(Dense(19, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        model.fit(
            X_train,
            y_train,
            batch_size=8,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        model.save(self.model_path)


if __name__ == "__main__":
    current_file = os.path.abspath(os.path.dirname(__file__))
    print(current_file)
    csv_filename = os.path.join(current_file, "../../Data/labeled_steam_reviews.csv")
    model_path = os.path.join(current_file, "keras_models/reviews_topic_model")
    joblib_path = os.path.join(
        current_file, "joblib_classes/reviews_topic_model.joblib"
    )

    data = pd.read_csv(current_file)
    model = ReviewsTopicModel(model_path)
    model.create_model(data)
    joblib.dump(model, joblib_path, compress=3)

    print("Model's dump is ready")
