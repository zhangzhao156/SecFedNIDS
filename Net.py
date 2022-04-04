# -*- coding: utf-8 -*-
# @Time    : 2021/7/3 10:17
# @Author  : zhao
# @File    : Net.py

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4,stride=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4, stride=2)
        )
        self.fn1 = torch.nn.Sequential(
            torch.nn.Linear(480, 128),
            torch.nn.ReLU(inplace=True)
        )
        self.fn2 = torch.nn.Sequential(
            torch.nn.Linear(128, 7),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fn1(x)
        x = self.fn2(x)
        return x

class CNN_UNSW(torch.nn.Module):
    def __init__(self):
        super(CNN_UNSW,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4,stride=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=4, stride=2)
        )
        self.fn1 = torch.nn.Sequential(
            torch.nn.Linear(288, 128),
            torch.nn.ReLU(inplace=True)
        )
        self.fn2 = torch.nn.Sequential(
            torch.nn.Linear(128, 2),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fn1(x)
        x = self.fn2(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fn1 = torch.nn.Sequential(
            torch.nn.Linear(68, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )
        self.fn2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )
        self.fn3 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )
        self.fn4 = torch.nn.Sequential(
            torch.nn.Linear(32, 7),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.fn1(x)
        x = self.fn2(x)
        x = self.fn3(x)
        x = self.fn4(x)
        return x


class MLP_UNSW(torch.nn.Module):
    def __init__(self):
        super(MLP_UNSW,self).__init__()
        self.fn1 = torch.nn.Sequential(
            torch.nn.Linear(42, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )
        self.fn2 = torch.nn.Sequential(
            torch.nn.Linear(128, 96),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )
        self.fn3 = torch.nn.Sequential(
            torch.nn.Linear(96, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )
        self.fn4 = torch.nn.Sequential(
            torch.nn.Linear(64, 2),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.fn1(x)
        x = self.fn2(x)
        x = self.fn3(x)
        x = self.fn4(x)
        return x


