from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import numpy as np


trainSet = datasets.MNIST(root="./MNIST", download=True,transform=ToTensor())
testSet = datasets.MNIST(root="./MNIST", download=True,transform=ToTensor())

img,label = trainSet[0]

