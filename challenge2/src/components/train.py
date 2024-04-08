import pandas as pd
import torch
import os
import json

from kfp import dsl
from torch.utils.data import DataLoader
from challenge2.src.configs import DiamondsDatasetConfig
from challenge2.src.utils.datasets import DiamondsPytorchDataset
from challenge2.src.utils.models import RegressionModel


def training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    train_dataset_length,
    val_dataset_length,
    checkpoints_path,
):
    num_epochs = 50
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, y in train_loader:
            optimizer.zero_grad()
            pred = model(x_cat, x_num)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_cat.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                pred = model(x_cat, x_num)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x_cat.size(0)

        train_loss = train_loss / train_dataset_length
        val_loss = val_loss / val_dataset_length

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1

        save_dir = os.path.join(checkpoints_path, f'epoch_{epoch + 1}')
        torch.save(model.state_dict, os.path.join(save_dir, 'weights.pth'))
        metadata = {
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as mfile:
            json.dump(metadata, mfile)

        print(f'Epoch {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

    return best_epoch


def get_checkpoints_path():
    checkpoints_path_template = 'challenge2/checkpoints/run_{}'
    n = 1
    path = checkpoints_path_template.format(n)
    while os.path.exists(path):
        n += 1
        path = checkpoints_path_template.format(n)
    return path


@dsl.component(base_image='python:3.8')
def pytorch_model_train_component(train_dataset_path: str, val_dataset_path: str) -> str:
    checkpoints_path = get_checkpoints_path()
    train_dataset = pd.read_csv(train_dataset_path) 
    val_dataset = pd.read_csv(val_dataset_path) 
    cat_features = list(filter(
        lambda x: x not in DiamondsDatasetConfig.NUMERICAL_FEATURES + [DiamondsDatasetConfig.TARGET],
        train_dataset.columns
    ))
    train_dataset = DiamondsPytorchDataset(
        train_dataset,
        cat_features,
        DiamondsDatasetConfig.NUMERICAL_FEATURES,
        DiamondsDatasetConfig.TARGET
    )
    val_dataset = DiamondsPytorchDataset(
        train_dataset,
        cat_features,
        DiamondsDatasetConfig.NUMERICAL_FEATURES,
        DiamondsDatasetConfig.TARGET
    )

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

    embedding_dim = 3
    model = RegressionModel(
        len(cat_features),
        embedding_dim,
        len(DiamondsDatasetConfig.NUMERICAL_FEATURES)
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("[*] Starting training loop")

    best_epoch = training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        len(train_dataset),
        len(val_dataset),
        checkpoints_path,
    )

    return os.path.join(checkpoints_path, f'epoch_{best_epoch}', 'weights.pth')
