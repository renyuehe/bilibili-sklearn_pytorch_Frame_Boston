import os
import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import DataLoader

from net import Net01
import dataset
import val

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = Net01().to(device) # 全连接网络
    # 优化器
    opt = optim.Adam(net.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    # 损失函数
    mseLoss = nn.MSELoss(reduction="mean")
    # 加载预训练权重
    netFilename = r"net_epoch_980_r2_0.78.pth"
    net.load_state_dict(torch.load(netFilename)) if os.path.exists(netFilename) else ...


    # 训练集 和 测试集
    train_datasets = dataset.BostonDataset(datafile="boston.xls", isTrain=True)
    test_datasets = dataset.BostonDataset(datafile="boston.xls", isTrain=False)

    tarin_dataloader = DataLoader(train_datasets, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=50, shuffle=True)

    # 训练
    for epoch in range(10000):
        net.train() # 开启训练
        train_losses = []
        for i, (data, target) in enumerate(tarin_dataloader):
            data, target = data.to(device), target.to(device)
            pred = net(data)
            loss = mseLoss(pred, target)

            # 梯度清空 反向传播 更新梯度
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss)

        train_losses = torch.tensor(train_losses)
        train_mean_loss = torch.mean(train_losses)
        # print("train loss = ", train_mean_loss.detach().cpu().item())

        if epoch % 500 == 0:
            print()
            net.eval()
            test_losses = []
            mse_s, r2_s, evs_s = [], [], []
            for i, (data, target) in enumerate(test_dataloader):
                data, target = data.to(device), target.to(device)
                pred = net(data)
                loss = mseLoss(pred, target)

                target, pred = target.detach().cpu(), pred.detach().cpu()
                mse, r2, evs= val.reg_calculate(target, pred)
                mse_s.append(mse)
                r2_s.append(r2)
                evs_s.append(evs)

                test_losses.append(loss)

            test_losses = torch.tensor(test_losses)
            test_mean_loss = torch.mean(test_losses)
            print("-----test loss= ", test_mean_loss.detach().cpu().item())

            mse_s, r2_s, evs_s = torch.tensor(mse_s), torch.tensor(r2_s), torch.tensor(evs_s)
            mse_mean, r2_mean, evs_mean = torch.mean(mse_s), torch.mean(r2_s), torch.mean(evs_s)
            mse_mean, r2_mean, evs_mean  = mse_mean.detach().cpu().item(), r2_mean.detach().cpu().item(), evs_mean.detach().cpu().item()


            print("-----mse:{0}-----r2:{1}-----evs:{2}".format(mse_mean, r2_mean, evs_mean))

            # torch.save(net.state_dict(), "net_epoch_{}_mse_{}_r2_{}.pth".format(epoch, round(mse_mean,2), round(r2_mean,2)))