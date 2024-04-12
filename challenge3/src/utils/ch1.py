import pandas as pd
import torch
from challenge3.src.utils.model import _predict


CH1_MODEL_PATH = 'challenge1/best_model.pt'
CH1_METADATA_PATH = 'challenge1/preproc_metadata.json'


def predict(user_data: pd.DataFrame):
    model = torch.jit.load(CH1_MODEL_PATH)
    return _predict(model, user_data, CH1_METADATA_PATH)
