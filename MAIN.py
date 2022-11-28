import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# 生成0101模式的序列
def generateTrainTrue():
    trainTrue = torch.tensor([random.uniform(0, 0.2), random.uniform(0.8, 1.0),
                              random.uniform(0, 0.2), random.uniform(0.8, 1)])
    return trainTrue


# 生成噪声序列
def generateTrainFalse(seed=None):
    if seed is None:
        trainFalse = torch.rand(4)
    else:
        pass
    return trainFalse


# 数据可视化
def lineChart(yData):
    fig, ax = plt.subplots()
    ax.plot(yData, 'bo')
    plt.show()


class Discriminator(nn.Module):
    """
    鉴别器可以作为单独的分类器进行使用
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(4, 3), nn.Sigmoid(), nn.Linear(3, 1), nn.Sigmoid())
        self.lossFunc = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.1)
        # 输入数据集的数量
        self.dataNum = None
        # 训练世代
        self.epoch = None
        self.counter = 0
        self.progress = []

    def forward(self, dataInput):
        return self.linear(dataInput)

    def train(self, dataInput, target):
        outcome = self.forward(dataInput)
        loss = self.lossFunc(outcome, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.counter += 1
        self.progress.append(loss.item())

    def trainEpoch(self, epoch, dataNum, dataInputT, dataInputF):
        self.epoch = epoch
        self.dataNum = dataNum
        for i in range(epoch):
            for j in range(dataNum):
                self.train(dataInputT, torch.tensor([1.0]))
                self.train(dataInputF, torch.tensor([0.0]))

    def queryLoss(self, interval):
        """
        对鉴别器的损失进行可视化
        :param interval: 打印损失的间隔
        :return: 损失散点图
        """

        plotY = []
        plotX = []
        for i in range(self.counter):
            if i % interval == 0:
                plotY.append(self.progress[i])
                plotX.append(i)
        fig, ax = plt.subplots()
        ax.scatter(plotX, plotY)
        plt.show()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 3), nn.Sigmoid(), nn.Linear(3, 4), nn.Sigmoid())
        self.lossFunc = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []

    def forward(self, dataInput):
        return self.model(dataInput)

    def trainG(self, dataInput, target, D):
        """
        :param dataInput: 数据集
        :param target: 数据标签
        :param D: 鉴别器
        """
        gOutput = self.forward(dataInput)
        dOutput = D.forward(gOutput)
        loss = self.lossFunc(dOutput, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.counter += 1
        self.progress.append(loss.item())

    def queryLoss(self, interval):
        plotY = []
        plotX = []
        for i in range(self.counter):
            if i % interval == 0:
                plotY.append(self.progress[i])
                plotX.append(i)
        fig, ax = plt.subplots()
        ax.scatter(plotX, plotY)
        plt.show()


class Gan:
    def __init__(self):
        self.D = Discriminator()
        self.G = Generator()
        self.counter = 0
        self.progress = []

    def train(self):
        # 用真实数据训练鉴别器
        self.D.train(generateTrainTrue(), torch.tensor([1.0]))
        # 用生成器生成的数据训练鉴别器
        self.D.train(self.G.forward(torch.tensor([0.5])).detach(), torch.tensor([0.0]))
        # 训练生成器
        self.G.trainG(torch.tensor([0.5]), torch.tensor([1.0]), self.D)

    def trainEpoch(self, epoch=1, dataNum=1000):
        for i in range(epoch):
            for j in range(dataNum):
                self.train()


if __name__ == "__main__":
    # net = Discriminator()
    # net.trainEpoch(1, 2000, generateTrainTrue(), generateTrainFalse())
    # net.queryLoss(100)
    # gan = Gan()
    # gan.trainEpoch(dataNum=2000)
    model = torch.load("./model/model.pk1")
