from config import *
import os
import torch
import math
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

base_dir = os.path.join(settings.data.dir, "ml2022spring-hw2", "libriphone")


def preprocess():
    # 每个pt文件的shape都是[*, 39]
    pass


def test():
    split_path = os.path.join(base_dir, "test_split.txt")
    with open(split_path) as f:
        test_files = f.readlines()

    length = 0
    for test_file in test_files:
        data_path = os.path.join(base_dir, "feat", "test", test_file).strip() + ".pt"
        data = torch.load(data_path)
        length += data.shape[0]
        # print(data.shape)

    print(length)

    train_label_path = os.path.join(base_dir, "train_labels.txt")
    with open(train_label_path) as f:
        train_labels = f.readlines()

    for train_label in train_labels:
        train_label_elearr = train_label.split(" ")
        data_path = (
            os.path.join(base_dir, "feat", "train", train_label_elearr[0]).strip()
            + ".pt"
        )
        data = torch.load(data_path)
        print(
            "train shape: ", data.shape, " label length: ", len(train_label_elearr) - 1
        )
        break


if __name__ == "__main__":
    test()
