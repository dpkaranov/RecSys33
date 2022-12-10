from collections import Counter
from typing import Dict

import dill
import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender


class UserKnn():
    """Class for fit-perdict UserKNN model
       based on ItemKNN model from implicit.nearest_neighbours
    """
    # pylint: disable=too-many-instance-attributes
    # 12 is reasonable in this case.

    def __init__(self, model=ItemItemRecommender, N_users: int = 20):
        self.N_users = N_users
        self.is_fitted = False
        self.users_inv_mapping: Dict[int, int] = dict()
        self.users_mapping: Dict[int, int] = dict()
        self.items_inv_mapping: Dict[int, int] = dict()
        self.items_mapping: Dict[int, int] = dict()
        self.watched = None
        self.item_idf = None
        self.user_knn = model
        self.weights_matrix = None
        self.n = None
        self.mapper = None

    def get_mappings(self, train) -> None:
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def get_matrix(self, df: pd.DataFrame,
                   user_col: str = 'user_id',
                   item_col: str = 'item_id',
                   weight_col: str = None,
                   users_mapping: Dict[int, int] = None,
                   items_mapping: Dict[int, int] = None):

        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.coo_matrix((
            weights,
            (
                df[user_col].map(self.users_mapping.get),
                df[item_col].map(self.items_mapping.get)
            )
        ))

        self.watched = df.groupby(user_col).agg({item_col: list})
        return interaction_matrix

    def idf(self, n: int, x: float):
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame):
        item_idf = pd.DataFrame.from_dict(Counter(df['item_id'].values),
                                          orient='index', columns=[
                                          'doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(
            lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame):
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train,
                                              users_mapping=self.users_mapping,
                                              items_mapping=self.items_mapping)

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)
        self.is_fitted = True

    def _generate_recs_mapper(self, model: ItemItemRecommender,
                              user_mapping: Dict[int, int],
                              user_inv_mapping: Dict[int, int], N: int):
        def _recs_mapper(user):
            user_id = user_mapping[user]
            recs = model.similar_items(user_id, N=N)
            return [user_inv_mapping[user]
                    for user, _ in recs], [sim for _, sim in recs]
        return _recs_mapper

    def prepare(self) -> None:
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        self.mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users
        )

    def predict(self, test: pd.DataFrame, N_recs: int = 10):
        recs = pd.DataFrame({'user_id': test['user_id'].unique()})
        recs['sim_user_id'], recs['sim'] = zip(
            *recs['user_id'].map(self.mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = recs[~(recs['sim'] >= 1)]\
            .merge(self.watched, left_on=['sim_user_id'],
                   right_on=['user_id'], how='left')\
            .explode('item_id')\
            .sort_values(['user_id', 'sim'], ascending=False)\
            .drop_duplicates(['user_id', 'item_id'], keep='first')\
            .merge(self.item_idf, left_on='item_id',
                   right_on='index', how='left')

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)
        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        return recs[recs['rank'] <= N_recs][['user_id', 'item_id']]

    def save(self, file_path: str):
        model_dict = {
            "model": self.user_knn,
            "users_mapping": self.users_mapping,
            "users_inv_mapping": self.users_inv_mapping,
            "N": self.N_users,
            "item_idf": self.item_idf,
            "watched": self.watched
        }

        with open(file_path, "wb") as f:
            # bin_data = dill.dumps(model_dict)
            dill.dump(model_dict, f, -1)

    def load(self, file_path: str):
        with open(file_path, "rb") as f:
            model_dict = dill.load(f)
        self.user_knn = model_dict['model']
        self.users_mapping = model_dict['users_mapping']
        self.users_inv_mapping = model_dict['users_inv_mapping']
        self.N_users = model_dict['N']
        self.item_idf = model_dict['item_idf']
        self.watched = model_dict['watched']
        self.is_fitted = True
