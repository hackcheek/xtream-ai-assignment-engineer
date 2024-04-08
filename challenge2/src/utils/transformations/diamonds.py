import pandas as pd
import numpy as np

from challenge2.src.utils.transformations.base import BaseDataTransform
from challenge2.src.configs import DiamondsDatasetConfig
from challenge2.src.utils.common import random_split


class DiamondsDataTransform(BaseDataTransform, DiamondsDatasetConfig):

    @classmethod
    def preprocessing(cls, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Capture just the important features and target
        data = data[cls.TRAINING_FEATURES + [cls.TARGET]]
        
        # Price should be more than zero
        data = data.loc[data['price'] > 0]
        
        # Sizes should be more than zero
        data = data.loc[data['x'] > 0]

        # Drop outliers
        data = cls.drop_outliers(data)

        # Drop unknown categorical labels
        data = cls.drop_unknown_labels(data)

        # Process variables for training
        for col_name in data.columns:
            if col_name in cls.CATEGORICAL_FEATURES:      
                data = cls.apply_one_hot_encoder(data, col_name)
            elif col_name in cls.NUMERICAL_FEATURES + [cls.TARGET]:
                data[col_name] = cls.apply_std_scaler(data[col_name])

        train_data, test_data, val_data = random_split(data, [0.6, 0.3, 0.1])
        return train_data, test_data, val_data


    @classmethod
    def postprocessing(cls, current_data: pd.DataFrame, user_data: pd.DataFrame) -> pd.DataFrame:
        target_std = user_data[cls.TARGET].std()
        target_mean = user_data[cls.TARGET].mean()
        col = current_data[cls.TARGET]
        col *= target_std
        col += target_mean
        current_data[cls.TARGET] = col
        return current_data
    

    @staticmethod
    def z_score(col):
        sigma = col.std()
        mean = col.median()
        return np.abs((col - mean) / sigma)


    @classmethod
    def drop_outliers(cls, data):
        threshold_4_cols = ['table', 'depth', 'carat']
        threshold_3_cols = ['x', 'y', 'z']
        for col_name in data.columns:
            if col_name in threshold_4_cols:
                data = data[cls.z_score(data[col_name]) < 4]
            elif col_name in threshold_3_cols:
                data = data[cls.z_score(data[col_name]) < 3]
        return data


    @classmethod
    def drop_unknown_labels(cls, data):
        return data[
            (data["clarity"].isin(cls.CLARITY_LABELS))
            & (data['color'].isin(cls.COLOR_LABELS))
            & (data['cut'].isin(cls.CUT_LABELS))
        ]


    @staticmethod
    def apply_one_hot_encoder(data, col_name):
        encoded_df = pd.get_dummies(data[col_name], prefix=col_name, dtype=int)
        return data.drop(columns=[col_name]).join(encoded_df)


    @staticmethod
    def apply_std_scaler(col):
        col -= col.mean()
        col /= col.std()
        return col
