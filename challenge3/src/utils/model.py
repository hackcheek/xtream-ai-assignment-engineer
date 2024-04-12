import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from challenge2.src.utils.datasets.diamonds import DiamondsPytorchDataset
from challenge2.src.utils.transformations.diamonds import DiamondsDataTransform
from challenge2.src.configs import DiamondsDatasetConfig


LABEL_PREDICTIONS = 'predictions'


def get_dataloader(data):
    d = data.copy()
    target = None
    if DiamondsDatasetConfig.TARGET in d.columns:
        d = d.drop(columns=[DiamondsDatasetConfig.TARGET])
        target = DiamondsDatasetConfig.TARGET

    cat_features = list(filter(
        lambda x: x not in DiamondsDatasetConfig.NUMERICAL_FEATURES + [DiamondsDatasetConfig.TARGET],
        d.columns
    ))

    d = DiamondsPytorchDataset(
        d,
        cat_features,
        DiamondsDatasetConfig.NUMERICAL_FEATURES,
        target
    )

    return DataLoader(d, batch_size=1024, shuffle=True)


def _predict(model, user_data: pd.DataFrame, metadata: dict | str | os.PathLike) -> pd.DataFrame:
    model.eval()

    test_data, _ = DiamondsDataTransform.preprocessing(user_data.copy(), for_inference=True)
    test_loader = get_dataloader(test_data)

    preds = []
    with torch.no_grad():
        preds = [model(x_cat, x_num) for x_cat, x_num in test_loader]

    preds = torch.cat(preds)
    preds = DiamondsDataTransform.postprocessing(preds, metadata)
    
    user_data[LABEL_PREDICTIONS] = preds
    return user_data


if __name__ == "__main__":
    model_path = 'challenge1/best_model.pt'
    metadata_path = 'challenge1/preproc_metadata.json'
    data_path = 'datasets/diamonds/new_records.csv'

    model = torch.jit.load(model_path)
    data = pd.read_csv(data_path)
    preds = _predict(model, data.copy(), metadata_path)
    print(preds)
