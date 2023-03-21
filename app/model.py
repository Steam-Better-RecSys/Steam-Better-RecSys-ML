import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    def __init__(self, d):
        self.data = d

    def predict(self, game_id: int, top: int = 5, offset: int = 0):
        selected_games_df = self.data.loc[self.data['App ID'].isin([game_id])].drop(columns=['App ID'])
        m_data = self.data.drop(columns=['App ID'])
        selected_games_union = pd.DataFrame(selected_games_df.max(axis=0)).T
        results = cosine_similarity(m_data.values, selected_games_union.values).T[0]
        return self.data.iloc[np.argpartition(results, -top)[-top:]]['App ID'].to_list()


if __name__ == "__main__":
    model = Model(pd.read_csv('../Data/Data.csv'))
    pickle.dump(model, open('model.pkl', 'wb'))

