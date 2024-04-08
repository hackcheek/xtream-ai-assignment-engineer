import pandas as pd

from kfp import dsl
from challenge2.src.utils.transformations.diamonds import DiamondsDataTransform


@dsl.component(base_image='python:3.8')
def preprocess_csv_component(user_data_path: str):
    print("[*] Preprocessing and Splitting data")

    print(f"{user_data_path=}")
    return 'hola desde preprocess'
    raise RuntimeError()

    train_set_path = 'challenge2/datasets/train_data_path.csv'
    test_set_path = 'challenge2/datasets/test_data_path.csv'
    val_set_path = 'challenge2/datasets/val_data_path.csv'

    user_data = pd.read_csv(user_data_path)
    train_data, test_data, val_data = DiamondsDataTransform.preprocessing(user_data)
    train_data.to_csv(train_set_path)
    test_data.to_csv(test_set_path)
    val_data.to_csv(val_set_path)
    return train_set_path, test_set_path, val_set_path
