import os
import zipfile
from urllib.parse import urlencode

import requests


def check_local_data() -> bool:
    return len(os.listdir('service/data')) == 0


def download_data() -> None:
    url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/vcIewxsvwHca3w'
    # Получаем загрузочную ссылку
    final_url = url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Загружаем файл и сохраняем его
    download_response = requests.get(download_url)
    with open('service/data/data.zip', 'wb') as f:
        f.write(download_response.content)

    with zipfile.ZipFile('service/data/data.zip', 'r') as zip_ref:
        zip_ref.extractall('service/data')
