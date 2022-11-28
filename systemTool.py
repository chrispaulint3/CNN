import numpy as np
import random
import matplotlib.pyplot as plt
import torch


# use matplotlib draw scatter plot
def plotLoss(interval, counterPara, progressPara, epoch, ylim=None):
    plotX = []
    plotY = []
    for i in range(counterPara):
        if i % interval == 0:
            plotX.append(i)
            plotY.append(progressPara[i])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x=plotX, y=plotY, marker=".")
    ax.set_facecolor("#F5F4EF")
    # 设置横纵轴坐标标签
    ax.set_xlabel("train times", fontfamily="serif", fontweight="semibold",
                  fontsize=15, color="#323232")
    ax.set_ylabel("loss", fontfamily="serif", fontweight="semibold",
                  fontsize=15, color="#323232")
    # 设置y轴限度
    ax.set_ylim(top=ylim)
    # 设置标题
    ax.set_title("train time and loss change", fontfamily="serif", fontweight="bold",
                 fontsize=15, color='#323232')  # ax.test()
    # 设置刻度
    ax.set_xticks(np.linspace(counterPara / epoch, counterPara, epoch))
    ax.tick_params(labelsize=12)
    # 设置水平参考线
    ax.axhline(y=0.25, c="black")
    plt.show()


def plotBar():
    pass


def testNet(net, testSet):
    counter = 0
    for img, label in testSet:
        oneHotResult = np.array(net.forward(img.reshape((28 * 28,))).detach())
        result = np.argmax(oneHotResult)
        if result == label:
            counter += 1
    return counter / len(testSet)

def deviceInfo():
    device = "cuda is available" if torch.cuda.is_available() else "cpu"
    print("using {} device".format(device))
    device = torch.device("cuda:0")
    return device


if __name__ == "__main__":
    random.seed(1)
    progress = np.array([random.random() for i in range(600)])
    print(progress)
    plotLoss(10, 600, progress, 5)
