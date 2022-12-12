import datetime

import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
from rectools import Columns
from rectools.metrics import (
    MAP,
    MeanInvUserFreq,
    Precision,
    Recall,
    Serendipity,
    calc_metrics,
)
from rectools.model_selection import TimeRangeSplit

from service.knn.userknn import UserKnn

metrics = {
    "prec@10": Precision(k=10),
    "recall@10": Recall(k=10),
    "map@10": MAP(k=10),
    "novelty": MeanInvUserFreq(k=10),
    "serendipity": Serendipity(k=10),
}

interactions = pd.read_csv('service/data/interactions.csv')

interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                             'total_dur': Columns.Weight},
                    inplace=True)
interactions['datetime'] = pd.to_datetime(interactions['datetime'])

# Кусок кода, позволяющий выбрат данные за месяц, два и т.д.
start = datetime.datetime(2021, 8, 12)
end = datetime.datetime(2021, 8, 22)
date_mask = (interactions['datetime'] >= start) & (
    interactions['datetime'] <= end)
dates = interactions['datetime'][date_mask]
interactions = interactions.iloc[dates.index]

max_date = interactions['datetime'].max()
min_date = interactions['datetime'].min()
print(f"min date in interactions: {min_date}")
print(f"max date in interactions: {max_date}")
print(f"number of days in data: {(max_date - min_date).days}")

# Задание параметров кросс-валидации
n_folds = 1
unit = "D"
n_units = 5

last_date = interactions[Columns.Datetime].max().normalize()
start_date = last_date - pd.Timedelta(n_folds * n_units + 1, unit=unit)
print(f"Start date and last date of the test fold: {start_date, last_date}")

periods = n_folds + 1
freq = f"{n_units}{unit}"
print(
    f"start_date: {start_date}\n"
    f"last_date: {last_date}\n"
    f"periods: {periods}\n"
    f"freq: {freq}\n"
)

date_range = pd.date_range(
    start=start_date, periods=periods, freq=freq, tz=last_date.tz)
print(f"Test fold borders: {date_range.values.astype('datetime64[D]')}")

# generator of folds
cv = TimeRangeSplit(
    date_range=date_range,
    filter_already_seen=True,
    filter_cold_items=True,
    filter_cold_users=True,
)
print(f"Real number of folds: {cv.get_n_splits(interactions)}")

results = []
fold_iterator = cv.split(interactions, collect_fold_stats=True)

for i_fold, (train_ids, test_ids, fold_info) in enumerate(fold_iterator):
    print(f"\n==================== Fold {i_fold}")
    print(fold_info)

    df_train = interactions.iloc[train_ids].copy()
    df_test = interactions.iloc[test_ids][Columns.UserItem].copy()

    catalog = df_train[Columns.Item].unique()

    model = UserKnn(model=TFIDFRecommender(), N_users=20)
    model.fit(df_train)

    recos = model.predict(df_test)

    metric_values = calc_metrics(
        metrics,
        reco=recos,
        interactions=df_test,
        prev_interactions=df_train,
        catalog=catalog,
    )

    fold = {"fold": i_fold, "model": model}
    fold.update(metric_values)
    print(fold)
    results.append(fold)

    if i_fold == n_folds:
        model.save('tfidf.dill')
