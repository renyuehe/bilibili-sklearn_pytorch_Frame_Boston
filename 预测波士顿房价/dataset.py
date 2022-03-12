import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle

from torch.utils.data import DataLoader, Dataset
class BostonDataset(Dataset):
    def __init__(self, datafile:str="boston.xls", isTrain:bool=True, test_size:float=0.3):
        # 加载所有的数据
        pd_data = pd.read_excel(datafile)
        pd_data = pd_data.sample(frac=1).reset_index(drop=True)# 打乱数据、重置索引
        counts = pd_data.shape[0]
        train_len = int(counts * (1-test_size))

        train_data = pd_data[:train_len]
        test_data = pd_data[train_len:]

        pd_data =  pd_data[:train_len] if isTrain else pd_data[train_len:]



        self.x = pd_data.iloc[:, 1:14].to_numpy()
        y = pd_data.iloc[:, 14].to_numpy()
        self.y = np.expand_dims(y, axis=1) if y.ndim == 1 else y

    def __len__(self):
        # 返回数据集长度
        return self.y.shape[0]

    def __getitem__(self, item):
        # for dataloader
        data = torch.from_numpy(self.x[item]).float()
        target = torch.from_numpy(self.y[item]).float()
        return data, target

if __name__ == '__main__':
    train_dataset = BostonDataset(datafile="boston.xls", isTrain=True)
    test_dataset = BostonDataset(datafile="boston.xls", isTrain=False)

    print(train_dataset[0])
    print(test_dataset[0])
    # from torch.utils.data import DataLoader
    # train_dataloader = DataLoader(train_dataset)
    # test_dataloader = DataLoader(test_dataset)


