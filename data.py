import logging
import os

os.environ['KAGGLE_CONFIG_DIR'] = './'

from kaggle.api.kaggle_api_extended import KaggleApi

def download_data(path: str) -> str:
    api = KaggleApi()
    api.authenticate()

    dataset = 'barelydedicated/airbnb-duplicate-image-detection'
    logging.info('Downloading dataset %s to %s', dataset, path)
    api.dataset_download_files(dataset, path=path, unzip=True)
    logging.info('Dataset %s downloaded to %s', dataset, path)

    return os.path.dirname(path)
