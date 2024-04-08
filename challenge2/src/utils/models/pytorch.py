import torch


class RegressionModel(torch.nn.Module):
    def __init__(self, cat_features_amount, embedding_dim, num_features_amount):
        super(RegressionModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=cat_features_amount,
            embedding_dim=embedding_dim
        )
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.dense1 = torch.nn.Linear(
            (embedding_dim * cat_features_amount) + num_features_amount,
            512
        )
        self.dense2 = torch.nn.Linear(512, 1024)
        self.dense3 = torch.nn.Linear(1024, 512)
        self.dense4 = torch.nn.Linear(512, 128)
        self.out = torch.nn.Linear(128, 1)

    def forward(self, x_cat, x_num):
        x_cat = self.embedding(x_cat)
        x_cat = x_cat.view(x_cat.size(0), -1)
        x = torch.cat([x_cat, x_num], dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.sig(x)
        x = self.out(x)
        return x
