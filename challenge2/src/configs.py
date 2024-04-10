class Config:
    @classmethod
    def asdict(cls) -> dict:
        return {
            i:j 
            for i, j in vars(cls).items()
            if not i.startswith('__') 
                and not i.endswith('__') 
                and not isinstance(j, classmethod)
        }


class DiamondsDatasetConfig(Config):
    BASELINE_DATASET_PATH: str = "challenge2/src/datasets/actual_dataset.csv"
    TARGET: str = "price"
    NUMERICAL_FEATURES: list[str] = ["x", "carat"]
    CATEGORICAL_FEATURES: list[str] = ["color", "clarity", "cut"]
    TRAINING_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    COLOR_LABELS: list[str] = ['H', 'I', 'F', 'G', 'E', 'D', 'J']
    CUT_LABELS: list[str] = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
    CLARITY_LABELS: list[str] = ['SI2', 'SI1', 'VS2', 'IF', 'VVS2', 'VS1', 'I1', 'VVS1']


class MinioConfig(Config):
    HOST = 'localhost'
    PORT = 9000
    ACCESS_KEY = 'minio'
    SECRET_KEY = 'minio123'
