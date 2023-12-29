from config import *
import os
import torch
import math
import pandas as pd
import numpy as np
from typing import Any
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# dataset = MyDataset(file)
# dataloader = DataLoader(dataset, batch_size, shuffle)
class Covid19Dataset(Dataset):
    def __init__(self, data, isTest=False) -> None:
        super().__init__()
        self.isTest = isTest
        if self.isTest:
            self.datax = torch.FloatTensor(data)
            self.datay = None
        else:
            self.datax = torch.FloatTensor(data[:, :-1])
            self.datay = torch.FloatTensor(data[:, -1])

    def __getitem__(self, index) -> Any:
        if self.isTest:
            return self.datax[index]
        else:
            return self.datax[index], self.datay[index]

    def __len__(self):
        return len(self.datax)

    def dim(self):
        return len(self.datax[0, :])


class Covid19Model(nn.Module):
    def __init__(self, in_dim):
        super(Covid19Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return torch.squeeze(self.network(x))


def train(epoches, learn_rate):
    df_train = pd.read_csv(settings.data.dir + "covid.train.csv")
    data_train, data_valid = random_split(df_train.values, [0.8, 0.2])
    train_dataset = Covid19Dataset(np.array(data_train))
    valid_dataset = Covid19Dataset(np.array(data_valid))

    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    model = Covid19Model(train_dataset.dim())

    usegpu = False
    if torch.cuda.is_available():
        usegpu = True
    if usegpu:
        model.to("cuda")

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), learn_rate)
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)

    best_loss = math.inf
    for epoch in range(epoches):
        train_loss = []
        model.train()  # 训练模式
        for x, y in train_loader:  # 每个batch
            optimizer.zero_grad()  # 清空上次的梯度信息
            if usegpu:
                x, y = x.to("cuda"), y.to("cuda")
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            train_loss.append(loss.detach().item())

        valid_loss = []
        model.eval()  # 评估模式
        for x_valid, y_valid in valid_loader:
            if usegpu:
                x_valid, y_valid = x_valid.to("cuda"), y_valid.to("cuda")
            with torch.no_grad():
                pred = model(x_valid)
                loss = criterion(pred, y_valid)
                valid_loss.append(loss.detach().item())
        avg_valid_loss = sum(train_loss) / len(train_loss)
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(settings.model.dir, "covid19.ckpt"),
            )
        print(
            "epoch: ",
            epoch,
            " train_loss: ",
            avg_valid_loss,
            " valid_loss: ",
            sum(valid_loss) / len(valid_loss),
        )


def test():
    df_test = pd.read_csv(settings.data.dir + "covid.test.csv")
    test_dataset = Covid19Dataset(df_test.values, True)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True)
    model = Covid19Model(test_dataset.dim())

    covid_ckpt = torch.load(os.path.join(settings.model.dir, "covid19.ckpt"))
    model.load_state_dict(covid_ckpt)

    test_loss = []
    for x_test in test_loader:
        with torch.no_grad():
            pred = model(x_test)
            test_loss.append(pred.detach().item())
    # print(max(test_loss), min(test_loss))
    df_out = pd.DataFrame(test_loss)
    df_out.columns = ["tested_positive"]
    df_out.to_csv(settings.data.dir + "output.csv")


if __name__ == "__main__":
    # 训练
    train(1, 1e-4)

    # 测试
    # test()
