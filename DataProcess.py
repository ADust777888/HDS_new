import torch
import os
import numpy as np
import pandas as pd
from torch.utils import data
from SMOTE import Smote
from config import config
from sklearn.model_selection import train_test_split

eps = 1e-8


class sampleData(data.Dataset):
    def __init__(self, FILE_NAME):
        '''
        myData_ = pd.read_csv(FILE_NAME, header=None)

        print(myData_.describe())
        print(myData_.info())
        myData_ = myData_.to_numpy()
        myData_ = myData_[:1000, :]
        print(myData_.shape)


        X = myData_[:, :-1]
        y = myData_[:, -1].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        #print(type(X_train))
        self.train_data = X_train
        self.train_label = y_train
        self.test_data = X_test
        self.test_label = y_test
        self.train_data = (self.train_data - self.train_data.mean(axis=0, keepdims=True)) \
                          / (self.train_data.std(axis=0, keepdims=True) + eps)
        self.test_data = (self.test_data - self.test_data.mean(axis=0, keepdims=True)) \
                         / (self.test_data.std(axis=0, keepdims=True) + eps)
        '''

        Data = pd.read_csv(FILE_NAME, header=None).to_numpy()
        Data = Data[:2000,:]
        np.random.shuffle(Data)
        Pos = Data[Data[:, -1] == 1].astype(np.float32)
        Neg = Data[Data[:, -1] == 0].astype(np.float32)

        Pos_len = Pos.shape[0]
        Neg_len = Neg.shape[0]
        # print(Pos_len)
        # print(Neg_len)
        if (Pos_len < Neg_len):
            tmp = Pos[:, :-1]
            g = int(Neg_len / Pos_len)
            tmp = Smote(tmp, N=g).over_sampling()
            len1 = tmp.shape[0]
            len2 = tmp.shape[1]
            kk = np.zeros((len1, len2 + 1))
            for i in range(len1):
                for j in range(len2):
                    kk[i][j] = tmp[i][j]
                kk[i][len2] = 1
            Pos = np.r_[Pos, kk]
        else:
            g = int(Pos_len / Neg_len)
            tmp = Neg[:, :-1]
            tmp = Smote(tmp, N=g).over_sampling()
            len1 = tmp.shape[0]
            len2 = tmp.shape[1]
            kk = np.zeros((len1, len2 + 1))
            for i in range(len1):
                for j in range(len2):
                    kk[i][j] = tmp[i][j]
                kk[i][len2] = 0
            Neg = np.r_[Neg, kk]
        Pos = Pos.astype(np.float32)
        Neg = Neg.astype(np.float32)

        self.num = Pos.shape[0] + Neg.shape[0]
        # print('%.2f'%(max(Pos.shape[0], Neg.shape[0]) / min(Pos.shape[0], Neg.shape[0])))
        d1 = int(Pos.shape[0] * 0.8)
        d2 = int(Neg.shape[0] * 0.8)

        # train_data = np.r_[Pos[:d1, :-1], Neg[:d2, :-1]]
        # train_label = np.r_[Pos[:d1, -1], Neg[:d2, -1]]
        # tomekLinks = SMOTETomek(random_state=0)
        # X_kos, y_kos = tomekLinks.fit_resample(train_data, train_label)

        # self.train_data = X_kos
        # self.train_label = y_kos
        self.train_data = np.r_[Pos[:d1, :-1], Neg[:d2, :-1]]
        self.train_label = np.r_[Pos[:d1, -1], Neg[:d2, -1]]
        self.test_data = np.r_[Pos[d1:, :-1], Neg[d2:, :-1]]
        self.test_label = np.r_[Pos[d1:, -1], Neg[d2:, -1]]
        self.train_data = (self.train_data - self.train_data.mean(axis=0, keepdims=True)) \
                          / (self.train_data.std(axis=0, keepdims=True) + eps)
        self.test_data = (self.test_data - self.test_data.mean(axis=0, keepdims=True)) \
                         / (self.test_data.std(axis=0, keepdims=True) + eps)

    # print(self.test_data, self.test_label)



    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]

    def __len__(self):
        return len(self.train_data)
