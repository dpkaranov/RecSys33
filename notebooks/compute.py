import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import dill
import pandas as pd

from service.knn.userknn import UserKnn

# место хранения результатов
result_data = {}

TRAIN = pd.read_csv(
    "interactions.csv",
    parse_dates=["last_watch_dt"]
)

# функция, позволяющая выполнить предсказание для одного пользователя
def get_recos(user_id):
    if user_id in model.users_mapping:
        data = TRAIN.loc[TRAIN.user_id.isin([user_id])]
        recos = list(model.predict(data)['item_id'])
    else:
        recos = []
    result_data.update({user_id: recos})


# Создаем парсер параметров запуска,
# чтобы иметь возможность выбрать модель и номер части данных
parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--users_part')
args = parser.parse_args()

# определяем часть данных
num_users = int(840196 / 4)
start = (int(args.users_part) - 1) * num_users
end = int(args.users_part) * num_users
users = pd.read_csv('users.csv')
users = list(users['user_id'])[start:end]
print(len(users))
print(start, end)

# выбираем модель
model_arg = args.model
if model_arg == 'tfidf':
    model = UserKnn()
    model.load('tfidf.dill')
    model.prepare()
if model_arg == 'bm25':
    model = UserKnn()
    model.load('bm25.dill')
    model.prepare()
if model_arg == 'cosine':
    model = UserKnn()
    model.load('cosine.dill')
    model.prepare()

print('started')
ts = time.time()

# распараллеливаем вычисления
with ThreadPoolExecutor(8) as executor:
    finished = 0
    for i in executor.map(get_recos, users):
        finished += 1
        print("COMPUTING: {}/{}".format(finished, num_users),
              end="\r", flush=True)

tf = time.time() - ts

print("Computed for {} seconds".format(round(tf)))

print("Saving data")
file_name = model_arg + str(args.users_part) + "_offline.dill"
with open(file_name, "wb") as f:
    dill.dump(result_data, f, -1)
print("Saved")
