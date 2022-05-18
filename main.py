import torch
import os
import DNN
import numpy as np
import pandas as pd
import time
from torch.utils import data
from config import config
from DNN import Loss
from DataProcess import sampleData
from torch import nn
from torch import optim
from FeatureSelection import selection
from FeatureRanking import Rank
from FuzzyInference import Fuzzy
from Eva import Eval
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


eps = 1e-8

start = time.perf_counter()
#print("开始运行")
FILE_NAME = config.FILE_NAME

dataset = sampleData(FILE_NAME)
dataloader = data.DataLoader(dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=0, drop_last=False)

#print("数据加载完成")
mid = time.perf_counter()
#print("加载数据消耗时间为:", mid - start)
#print("开始特征选择")
selectFeature, w = selection(dataset.train_data, dataset.train_label)

#print("selectFeature:")
#print(selectFeature)

#print("w:")
#print(w)

mid2 = time.perf_counter()
#print("特征选择消耗时间为:", mid2 - mid)

X_train = dataset.train_data[:, selectFeature]
y_train = dataset.train_label

#print("开始计算FR")
FR = Rank(X_train, y_train, w)
#print("FR:")
#print(FR)
mid3 = time.perf_counter()
#print("计算FR耗时:", mid3 - mid2)

X_test = dataset.test_data[:, selectFeature]
y_test = dataset.test_label

n = len(w)
'''
num = X_test.shape[0]
Pnormal = []
Pinfected = []
for i in range(num):
    Pnormali, Pinfectedi = Fuzzy(n, X_test[i], FR)
    Pnormal.append(Pnormali)
    Pinfected.append(Pinfectedi)

p1 = np.array(Pnormal).reshape(-1, 1)
p2 = np.array(Pinfected).reshape(-1, 1)
P = np.c_[p1, p2]
'''

X_train = torch.from_numpy(X_train)
X_train = X_train.to(device)
y_train = torch.from_numpy(y_train)
y_train = y_train.to(device)
X_test = torch.from_numpy(X_test)
X_test = X_test.to(device)
y_test = torch.from_numpy(y_test)
y_test = y_test.to(device)

#model = DNN.Net(X_train.shape[1]).to(device)
#criterion = Loss
#optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0, weight_decay=0)

train_output = []
test_output = []

sAcc = 0
sG_mean = 0
sF1 = 0
sRecall = 0

print("开始训练模型:")
for j in range(10):
    model = DNN.Net(X_train.shape[1]).to(device)
    criterion = Loss
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0, weight_decay=0)
    for epoch in range(config.n_epoch):
        model.train()
        for X, y in dataloader:
            X = X[:, selectFeature].to(device)
            y = y.to(device)
            y_hat = model(X)
            y = y.to(torch.float32)
            y_hat = y_hat.to(torch.float32)
            loss = criterion(y=y, y_hat=y_hat)

            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()

            optimizer.step()

        model.eval()
        y_output_train = model(X_train)
        train_output.append(criterion(y=y_train, y_hat=y_output_train).item())
        y_output_test = model(X_test)
        test_output.append(criterion(y=y_test, y_hat=y_output_test).item())

    #print("训练完成:")
    mid4 = time.perf_counter()
    #print("训练耗时:", mid4 - mid3)

    #print("开始测试:")
    y_output = model(X_test)
    y_output = y_output.detach().cpu().numpy()

    Output = y_output
    # Output = (P + y_output) / 2
    # Output  = np.array(Output).tolist()
    ans = []
    N = len(Output)
    for i in range(N):
        if (Output[i] > 0.5):
            ans.append(1)
        else:
            ans.append(0)

    ans = np.array(ans)
    num_P, num_N, TP, TN, FP, FN, P, R, F1, Acc, G_mean, Recall = Eval(y_test, ans)
    sAcc += Acc
    sG_mean += G_mean
    sF1 += F1
    sRecall += Recall
    print(j, ':   ', 'Acc:%.10lf' % Acc, 'G_mean:%.10lf' % G_mean, 'F1:%.10lf' % F1, 'Recall:%.10lf' % Recall)
aveAcc = sAcc / 10
aveG_mean = sG_mean / 10
aveF1 = sF1 / 10
aveRecall = sRecall / 10
print("平均效果", ':  ', 'Acc:%.10lf' % aveAcc, 'G_mean:%.10lf' % aveG_mean, 'F1:%.10lf' % aveF1, 'F1:%.10lf' % aveRecall)
#end = time.perf_counter()
#print("测试完成:", end - mid4)
#print("总耗时为:", end - start)
