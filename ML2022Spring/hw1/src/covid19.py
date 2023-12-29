from config import *

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    # 使用gpu训练模型时，可能引入额外的随机源，使得结果不能准确再现
    # 为了能够使得结果可复现，我们需要固定随机源
    torch.backends.cudnn.deterministic = True

    # 会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    # 其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间
    torch.backends.cudnn.benchmark = False

    # 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进
    np.random.seed(seed)
    torch.manual_seed(seed)  # 设置cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置gpu


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(
        data_set,
        [train_set_size, valid_set_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
            pred = model(x)
            # detach() 用于返回一个新的 Tensor，这个 Tensor 和原来的 Tensor 共享相同的内存空间，
            # 但是不会被计算图所追踪，也就是说它不会参与反向传播，不会影响到原有的计算图
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def select_feature(train_data, valid_data, test_data, select_all=True):
    """Selects useful features to perform regression"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = (
        train_data[:, :-1],
        valid_data[:, :-1],
        test_data,
    )

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.

    return (
        raw_x_train[:, feat_idx],
        raw_x_valid[:, feat_idx],
        raw_x_test[:, feat_idx],
        y_train,
        y_valid,
    )


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(
        reduction="mean"
    )  # Define your loss function, do not modify this.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=0.9
    )

    writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir("./models"):
        os.mkdir("./models")  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config["n_epochs"], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        # leave 迭代结束后保留进度条
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            # item() 返回tensor中的值
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar("Loss/train", mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(
            f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}"
        )
        writer.add_scalar("Loss/valid", mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config["save_path"])  # Save your best model
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config["early_stop"]:
            print("\nModel is not improving, so we halt the training session.")
            return


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "seed": 5201314,  # Your seed number, you can pick your lucky number. :)
        "select_all": True,  # Whether to use all features.
        "valid_ratio": 0.2,  # validation_size = train_size * valid_ratio
        "n_epochs": 10,  # Number of epochs.
        "batch_size": 256,
        "learning_rate": 1e-5,
        "early_stop": 400,  # If model has not improved for this many consecutive epochs, stop training.
        "save_path": "./models/model.ckpt",  # Your model will be saved here.
    }

    # Set seed for reproducibility
    same_seed(config["seed"])

    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)
    train_data, test_data = (
        pd.read_csv(settings.data.dir + "ml2022spring-hw1\covid.train.csv").values,
        pd.read_csv(settings.data.dir + "ml2022spring-hw1\covid.test.csv").values,
    )
    train_data, valid_data = train_valid_split(
        train_data, config["valid_ratio"], config["seed"]
    )

    # Print out the data size.
    print(
        f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}"""
    )

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feature(
        train_data, valid_data, test_data, config["select_all"]
    )

    # Print out the number of features.
    print(f"number of features: {x_train.shape[1]}")

    train_dataset, valid_dataset, test_dataset = (
        COVID19Dataset(x_train, y_train),
        COVID19Dataset(x_valid, y_valid),
        COVID19Dataset(x_test),
    )

    # Pytorch data loader loads pytorch dataset into batches.
    # 设置pin_memory为True，意味着生成的Tensor所处的内存是“锁页内存”，这样转化到GPU的显存就会更快一些
    # 主机中的内存有锁页和不锁页两种方式，锁页内存存放的内容任何情况都不会与虚拟内存进行交换（虚拟内存就是硬盘）
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
    )

    model = My_Model(input_dim=x_train.shape[1]).to(
        device
    )  # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model, config, device)
