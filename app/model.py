import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class Model:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tags = self.df.columns
        self.tag_importance = self.__calculate_tag_importance()

    def __calculate_tag_importance(self):
        """Function that computes of importance of each tag based on how often it occurs"""
        tags_importance = {}
        for tag in self.tags:
            tags_importance[tag] = self.df.loc[self.df[tag] > 0][tag].count()
        return dict(reversed(sorted(tags_importance.items(), key=lambda x: x[1])))

    def __keep_important_features(self, df: pd.DataFrame, num_max_features: int):
        """Function to keep weighted tags below maximum number"""
        features = df.loc[:, df.columns[df.max(axis=0) > 0]].columns
        num_cur_features = len(features)
        for tag in self.tag_importance:
            if tag in features and num_cur_features > num_max_features:
                df[tag] = 0
                num_cur_features -= 1
            elif num_cur_features == 0:
                break
        return df

    def __get_most_important_features(self, df: pd.DataFrame, top_features: int = 1):
        """Function to get names of most important features from the vector"""
        features = df.loc[:, df.columns[df.max(axis=0) > 0]].columns
        top_tags = []
        for tag in reversed(self.tag_importance):
            if tag in features and len(top_tags) < top_features:
                top_tags.append(tag)
            elif len(top_tags) == top_features:
                break
        return top_tags

    def __get_clear_dataframe(self, df: pd.DataFrame):
        """Function to keep weights of tags below 2"""
        for tag in self.tags:
            if df[tag].item() > 1:
                df[tag] = 1
        return df

    def reduce_vector(self, vector: list):
        """Function for replacing a fractional vector with an integer one to reduce cookie size"""
        values = [i * 100 for i in vector]
        values = np.array(values, dtype="int16")
        return " ".join(map(str, values))

    def set_initial_vector(self, game_ids: list):
        """Function to set initial prediction vector with values from selected games"""
        max_features_total = 20
        max_features_per_game = 5
        selected_df = pd.DataFrame(0, columns=self.tags, index=[""])

        for game_id in game_ids:
            selected_df += self.__keep_important_features(
                self.df.loc[[int(game_id)]], max_features_per_game
            ).values

        return self.reduce_vector(
            self.__get_clear_dataframe(
                self.__keep_important_features(selected_df, max_features_total)
            ).values[0]
        )

    def predict(
        self,
        predicted_vector,
        game_id: int = 0,
        game_status: int = 1,
        top: int = 10,
        offset: int = 0,
    ):
        """Function to predict next results based on a predicted vector and a current game status:
        liked/disliked/ignored"""
        if predicted_vector is None:
            predicted_vector = np.zeros((1, len(self.tags)), dtype=float)
        else:
            predicted_vector = np.fromstring(predicted_vector, dtype=float, sep=" ")
        predicted_vector /= 100
        predicted_vector = [predicted_vector]
        predicted_dataframe = pd.DataFrame(predicted_vector, columns=self.tags)

        game_values = pd.DataFrame(0, columns=self.tags, index=[""]).values
        if game_id != 0 and game_status != 0:
            game_df = self.df.loc[[game_id]]
            features = 5
            game_values = (
                self.__keep_important_features(game_df, features).values * game_status
            )

        predicted_dataframe += game_values

        predicted_dataframe = self.__keep_important_features(predicted_dataframe, 20)

        predicted_values = predicted_dataframe.values

        most_important_tag = self.__get_most_important_features(predicted_dataframe)[0]

        results_df = self.df.copy()
        results_df = results_df[results_df[most_important_tag] > 0]
        results_df["distance"] = cosine_distances(results_df.values, predicted_values)
        results_df = results_df.sort_values(by=["distance"], ascending=True)

        return {
            "vector": self.reduce_vector(predicted_values[0]),
            "games": results_df.head(offset + top).tail(top).index.to_list(),
        }


if __name__ == "__main__":
    data = pd.read_csv("../Data/steam_wizzard_tags_data.csv", index_col=0)
    model = Model(data)
    joblib.dump(model, "model.joblib", compress=3)

    print("Model's dump is ready")
