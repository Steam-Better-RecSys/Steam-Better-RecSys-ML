import joblib
import os
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


if __name__ == "__main__":
    current_file = os.path.abspath(os.path.dirname(__file__))
    print(current_file)
    csv_filename = os.path.join(current_file, "../../Data/labeled_steam_reviews.csv")
    model_path = os.path.join(current_file, "keras_models/reviews_class_model")
    joblib_path = os.path.join(
        current_file, "joblib_classes/reviews_class_model.joblib"
    )

    data = pd.read_csv(csv_filename)
    model = ReviewsClassModel(model_path)
    model.create_model(data)
    joblib.dump(model, joblib_path, compress=3)

    print("Model's dump is ready")
