import torch
import torch.nn as nn
import torchvision
import task_generator


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            # 预期输入形状（1，1,28,28)
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            # (1,64,26,26)
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            # (1,64,13,13)
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            # (1,64,13,13)
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            # (1,64,11,11)
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
            # (1,64,5,5)
        self.layer3 = nn.Sequential(
            # (1,64,5,5)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # (1,64,3,3)
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            # (1,64,3,3)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # (1,64,1,1)
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


# load dataset
# train = torchvision.datasets.Omniglotle(root="./data", background=True, download=True)
# test = torchvision.datasets.Omniglot(root="./data", background=False, download=True)
# test convolution net, input images 28*28
img = torch.ones((1,1,13, 13))
b = img.view(13*13)
print(img.shape)
print(b.shape)