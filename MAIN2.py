import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import time
import systemTool

trainSet = datasets.MNIST(root="./MNIST", download=True, transform=ToTensor())
testSet = datasets.MNIST(root="./MNIST", download=True, transform=ToTensor())


def oneHotLabel(lab, num,device="cpu"):
    oneHot = torch.zeros(num,device=device)
    oneHot[lab] = 1.0
    return oneHot


def calcAccuracy():
    pass


class Discriminator(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(28 * 28, 200), nn.LeakyReLU(), nn.LayerNorm(200),
                                   nn.Linear(200, 10), nn.LeakyReLU())
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.lossFunc = nn.MSELoss()
        self.device = device
        self.counter = 0
        self.progress = []

    def forward(self, dataInput):
        return self.model(dataInput)

    def trainD(self, dataInput, label):
        result = self.forward(dataInput)
        loss = self.lossFunc(result, label)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.counter += 1
        self.progress.append(loss.item())

    def trainLoop(self, trainSet, epoch=5):
        for i in range(epoch):
            for img, label in trainSet:
                label = oneHotLabel(label, 10)
                self.trainD(img.reshape((28 * 28,)), label)


if __name__ == "__main__":
    start = time.time()
    net = Discriminator()
    net.trainLoop(trainSet=trainSet)
    end = time.time()
    print(end - start)
    systemTool.testNet(net,testSet)

