import random
import typing as tp
from itertools import cycle, islice

import dill
import pandas as pd
from pydantic import BaseModel

from service.data_load import check_local_data, download_data
from service.knn.userknn import UserKnn

if check_local_data():
    print("downloading data")
    download_data()
    print('loaded')

TRAIN = pd.read_csv(
    "./service/data/interactions.csv",
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


class UserKNNModelOnline(OurModels):
    def __init__(self, model_path) -> None:
        self.model = UserKnn()
        self.model.load(model_path)
        self.model.prepare()

        with open('service/data/popular.dill', "rb") as f:
            self.popular = dill.load(f)

    def _postproc(self, recs) -> list:
        if len(recs) == 0:
            return self.popular
        if len(recs) < 10:
            return list(set(recs) | set(self.popular))[:10]
        return recs

    def get_reco(self, user_id) -> list:
        data = TRAIN.loc[TRAIN['user_id'] == int(user_id)]
        if user_id in self.model.users_mapping:
            recs = list(self.model.predict(data)['item_id'])
            return self._postproc(recs)
        return self.popular


class UserKNNModelOffline(OurModels):
    def __init__(self, data_path) -> None:
        with open(data_path, "rb") as f:
            self.recs = dill.load(f)

        with open('service/data/popular.dill', "rb") as f:
            self.popular = dill.load(f)

    def get_reco(self, user_id) -> list:
        recs = self.recs.get(user_id)
        if recs is not None:
            return recs
        return self.popular


ALL_MODELS = {'first_try': FirstTry(),
              'popular_model': PopularModel(TRAIN),
              'userknn_model': UserKNNModelOnline(
    model_path='service/data/knn/tfidf.dill'),
    'userknn_model_offline': UserKNNModelOffline(
    data_path='service/data/knn/all_alg3.dill')}


def get_models() -> tp.Dict[str, OurModels]:
    return ALL_MODELS
