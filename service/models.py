import random
import typing as tp
from itertools import cycle, islice

import pandas as pd
from pydantic import BaseModel


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class PopularRecommender:
    def __init__(
        self, max_K=10, days=30, item_column="item_id", dt_column="date"
    ):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []

    def fit(self, df):
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.recommendations = (
            df.loc[df[self.dt_column] > min_date, self.item_column]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def recommend(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            return list(islice(cycle([recs]), len(users)))


class FirstTry:
    def __init__(self):
        pass

    def get_reco(self, user_id):
        return random.sample(range(1, 20), 10)


class PopularModel:
    def __init__(self):
        TRAIN = pd.read_csv(
            "./service/data/interactions.csv", parse_dates=["last_watch_dt"]
        )
        PMODEL = PopularRecommender(days=30, dt_column="last_watch_dt")
        PMODEL.fit(TRAIN)

        self.model = PMODEL

    def get_reco(self, user_id):
        return list(self.model.recommend(users=[user_id], N=10)[0])


MODELS = {"first_try": FirstTry, "popular_model": PopularModel}


def get_models():
    return MODELS
