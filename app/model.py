import joblib
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


class Model:
    def __init__(self, game_df: pd.DataFrame):
        self.df = game_df.set_index("App ID")
        self.tags = self.__get_tags_from_game_df()
        self.tag_importance = self.__calculate_tag_importance()

    def __get_tags_from_game_df(self):
        return self.df.columns

    def __calculate_tag_importance(self):
        tags_df = self.df
        tags_importance = {}
        for tag in tags_df.columns:
            tag_c = tags_df[tag]
            tags_importance[tag] = tag_c[tag_c > 0].count()
        return dict(reversed(sorted(tags_importance.items(), key=lambda x: x[1])))

    def __keep_important_features(self, df: pd.DataFrame, num_max_features: int):
        features = df.loc[:, df.columns[df.max(axis=0) > 0]].columns
        num_cur_features = len(features)
        for tag in self.tag_importance:
            if tag in features and num_cur_features > num_max_features:
                df[tag] = 0
                num_cur_features -= 1
        print(df.loc[:, df.columns[df.max(axis=0) > 0]].columns)
        return df

    def __get_clear_dataframe(self, df: pd.DataFrame):
        for tag in self.tag_importance:
            if df[tag].item() > 2:
                df[tag] = 2
        return df

    def set_initial_pool(self, game_ids: list):
        max_features = 8
        selected_df = pd.DataFrame(0, columns=self.tags, index=[''])

        for game_id in game_ids:
            game_df = self.df.loc[[game_id]]
            selected_df += self.__keep_important_features(game_df, max_features).values

        return self.__get_clear_dataframe(selected_df)

    def predict(self, predicted_vector, game_id: int = 0, liked: int = 1, top: int = 10, offset: int = 0, iteration: int = 1):
        if predicted_vector is not None:
            predicted_vector = np.array(literal_eval(predicted_vector))
        else:
            predicted_vector = np.zeros((1, len(self.tags)), dtype=float)
        predicted_dataframe = pd.DataFrame(predicted_vector, columns=self.tags)
        predicted_values = self.__keep_important_features(predicted_dataframe, 30).values

        game_values = pd.DataFrame(0, columns=self.tags, index=['']).values
        if game_id != 0:
            game_df = self.df.loc[[game_id]]
            features = min(5 + (iteration - 1), 10)
            game_values = self.__keep_important_features(game_df, features).values * liked

        predicted_values += game_values

        results_df = self.df.copy()
        results_df['distance'] = cosine_distances(self.df.values, predicted_values)
        results_df = results_df.sort_values(by=['distance'], ascending=True)

        return {
            'vector': np.array2string(predicted_values, precision=2, separator=',', suppress_small=True),
            'recs': results_df.head(offset + top).tail(top).index.to_list()
        }


if __name__ == "__main__":
    model = Model(pd.read_csv('../Data/Data.csv'))
    joblib.dump(model, "model.joblib", compress=3)

