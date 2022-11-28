import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import time


# 测试设备信息，使用cuda:0设备
def deviceInfo():
    device = "cuda is available" if torch.cuda.is_available() else "cpu"
    print("using {} device".format(device))
    device = torch.device("cuda:0")
    return device


# 训练集与测试集的加载
train_set = datasets.MNIST(root="./MNIST", download=True, transform=ToTensor())
test_set = datasets.MNIST(root="./MNIST", download=True, transform=ToTensor())


def generateTrainFalse(size):
    trainFalse = torch.rand(size)
    return trainFalse


def generateRandom(size, device="cpu"):
    rand = torch.rand(size,device=device)
    return rand


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(28 * 28, 200), nn.Sigmoid(), nn.Linear(200, 1), nn.Sigmoid())
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.lossFunc = nn.MSELoss()
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

    def trainLoop(self, trainSet):
        for img, label in trainSet:
            self.trainD(img.reshape((28 * 28,)), torch.tensor([1.0]))
            self.trainD(generateTrainFalse(28 * 28), torch.tensor([0.0]))

    def query_loss(self, interval):
        plotX = []
        plotY = []
        for i in range(self.counter):
            if self.counter % interval:
                plotX.append(i)
                plotY.append(self.progress[i])
        fig, ax = plt.subplots()
        ax.scatter()
        plt.show()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 200), nn.Sigmoid(), nn.Linear(200, 784), nn.Sigmoid())
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.lossFunc = nn.MSELoss()
        self.counter = 0
        self.progress = []

    def forward(self, dataInput):
        return self.model(dataInput)

    def trainG(self, dataInput, label, D):
        GOutput = self.forward(dataInput)
        DOutput = D.forward(GOutput)
        loss = self.lossFunc(DOutput, label)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.counter += 1
        self.progress = []


class GAN:
    def __init__(self, device="cpu"):
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)
        self.device = device

    def trainGan(self, trainSet, testSet):
        for data, label in trainSet:
            data = data.to(self.device)
            label = torch.tensor([label],device=self.device)
            self.D.trainD(data.reshape((28 * 28,)), torch.tensor([1.0], device=self.device))
            self.D.trainD(self.G.forward(generateRandom(1,device=self.device)).detach(),
                          torch.tensor([0.0], device=self.device))
            self.G.trainG(generateRandom(1,device=self.device), torch.tensor([1.0], device=self.device), self.D)


if __name__ == "__main__":
    dev = deviceInfo()
    # # 程序起始时间
    # startTime = time.time()
    # gan = GAN(device="cuda")
    # gan.trainGan(train_set, test_set)
    # # torch.save(gan, "./model/ganMNIST.pk1")
    # # 程序终止时间
    # endTime = time.time()
    # print("train model using: {} seconds".format(endTime - startTime))
    gan = torch.load("./model/ganMNIST.pk1")
    out = gan.G.forward(generateRandom(1))
    fig,ax = plt.subplots()
    ax.imshow(out.detach().numpy().reshape((28,28)))
    plt.show()
    print(out)

