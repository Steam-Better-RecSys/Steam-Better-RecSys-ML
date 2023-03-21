import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    def __init__(self, game_df: pd.DataFrame):
        self.df = game_df
        self.initial_vector = []

    def set_initial_pool(self, game_ids: list):
        selected_df = self.df.loc[self.df['App ID'].isin(game_ids)].drop(columns=['App ID'])
        self.initial_vector = selected_df.max(axis=0).values

    def predict(self, game_id: int, liked: bool = True, top: int = 5, offset: int = 0):
        data = self.df.set_index('App ID')
        selected_game = data.loc[[game_id]].max(axis=0).values

        m = 1 if liked else -1
        self.initial_vector = (selected_game * m)

        results = cosine_similarity(data.values, [self.initial_vector]).T[0]
        top = np.argpartition(results, -top)[-top:]
        return self.df.loc[top]['App ID'].to_list()


if __name__ == "__main__":
    model = Model(pd.read_csv('../Data/Data.csv'))
    joblib.dump(model, "model.joblib")

