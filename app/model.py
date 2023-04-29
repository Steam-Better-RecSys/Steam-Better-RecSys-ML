import joblib
import pandas as pd
import numpy as np
from ast import literal_eval
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
        return df

    def __get_clear_dataframe(self, df: pd.DataFrame):
        """Function to keep weights of tags below 2"""
        for tag in self.tags:
            if df[tag].item() > 2:
                df[tag] = 2
        return df

    def set_initial_vector(self, game_ids: list):
        """Function to set initial prediction vector with values from selected games"""
        max_features_total = 20
        max_features_per_game = 5
        selected_df = pd.DataFrame(0, columns=self.tags, index=[""])

        for game_id in game_ids:
            selected_df += self.__keep_important_features(
                self.df.loc[[int(game_id)]], max_features_per_game
            ).values

        return self.__get_clear_dataframe(
            self.__keep_important_features(selected_df, max_features_total)
        )

    def predict(
        self,
        predicted_vector,
        game_id: int = 0,
        game_status: int = 1,
        top: int = 10,
        offset: int = 0,
        iteration: int = 1,
    ):
        """Function to predict next results based on a predicted vector and a current game status:
        liked/disliked/ignored"""
        predicted_vector = np.array(literal_eval(predicted_vector))
        predicted_dataframe = pd.DataFrame(predicted_vector, columns=self.tags)
        predicted_values = self.__keep_important_features(
            predicted_dataframe, 15
        ).values

        game_values = pd.DataFrame(0, columns=self.tags, index=[""]).values
        if game_id != 0 and game_status != 0:
            game_df = self.df.loc[[game_id]]
            features = min(5 + (iteration - 1), 10)
            game_values = (
                self.__keep_important_features(game_df, features).values * game_status
            )

        predicted_values += game_values

        results_df = self.df.copy()
        results_df["distance"] = cosine_distances(self.df.values, predicted_values)
        results_df = results_df.sort_values(by=["distance"], ascending=True)

        return {
            "vector": np.array2string(
                predicted_values, precision=2, separator=",", suppress_small=True
            ),
            "games": results_df.head(offset + top).tail(top).index.to_list(),
        }


if __name__ == "__main__":
    data = pd.read_csv("../Data/steam_wizzard_tags_data.csv", index_col=0)
    model = Model(data)
    joblib.dump(model, "model.joblib", compress=3)

    print("Model's dump is ready")
