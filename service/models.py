import random
import typing as tp
from itertools import cycle, islice

import dill
import pandas as pd
from pydantic import BaseModel

TRAIN = pd.read_csv(
                "./data/interactions.csv",
                parse_dates=["last_watch_dt"]
)


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class PopularRecommender:
    def __init__(
        self, max_K=10, days=30, item_column="item_id", dt_column="date"
    ) -> None:
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations: tp.List[int] = []

    def fit(self, df) -> None:
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(
            days=self.days
        )
        self.recommendations = (
            df.loc[df[self.dt_column] > min_date, self.item_column]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def recommend(self, users=None, N=10) -> list:
        recs = self.recommendations[:N]
        if users is None:
            return recs
        return list(islice(cycle([recs]), len(users)))


class OurModels:
    def get_reco(self, user_id) -> list:
        pass


class FirstTry(OurModels):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id) -> list:
        return random.sample(range(1, 20), 10)


class PopularModel(OurModels):
    def __init__(self, train) -> None:
        self.pmodel = PopularRecommender(days=30, dt_column="last_watch_dt")
        self.pmodel.fit(train)

    def get_reco(self, user_id) -> list:
        return list(self.pmodel.recommend(users=[user_id], N=10)[0])


class LightFmOffline(OurModels):
    def __init__(self) -> None:
        with open('./models/lightfm_preds.dill', 'rb') as file:
            self.model = dill.load(file)
        self.submodel = PopularModel(TRAIN)
        self.popular_reco = self.submodel.get_reco(1)

    def get_reco(self, user_id) -> list:
        if user_id in self.model.keys():
            val = self.model[user_id]
            if len(val) < 10:
                val += self.popular_reco
                val = list(set(val))
        else:
            val = self.popular_reco
        return val[:10]


ALL_MODELS = {
                'first_try': FirstTry(),
                'popular_model': PopularModel(TRAIN),
                'lightfm_model': LightFmOffline()
}


def get_models() -> tp.Dict[str, OurModels]:
    return ALL_MODELS
