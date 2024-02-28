import joblib
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D

from text_classifier_model import TextClassifier


class ReviewsClassModel(TextClassifier):
    def create_model(self, df: pd.DataFrame):
        df = self.preprocess(df)

        X_train, X_test, y_train, y_test = self.prepare_data(df, "class", 50, 0.8)

        model = Sequential()
        model.add(Embedding(len(self.word_2_index), 100))
        model.add(SpatialDropout1D(0.8))
        model.add(Conv1D(filters=16, kernel_size=3, padding='same',
                             activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, dropout=0.8, recurrent_dropout=0.8))
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
    data = pd.read_csv("../Data/labeled_steam_reviews.csv")
    model = ReviewsClassModel("keras_models/reviews_class_model")
    model.create_model(data)
    joblib.dump(model, "model_reviews_class_model.joblib", compress=3)

    print("Model's dump is ready")
