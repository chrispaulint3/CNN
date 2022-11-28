import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt


def RGB_To_Grayscale(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return (R + G + B) / 3


def mean_filter(kSize):
    kernel = np.ones((kSize, kSize))
    return (1 / sum(kernel)) * kernel


# 均值滤波
def means_blur(image, kernel, stride):
    pass


def sobel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def sobel_y():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def convolutionFig(imgArr, kernel):
    convolutionRes = np.zeros(imgArr.shape)
    print(convolutionRes)
    kernelSize = kernel.shape[0]
    padding = int((kernelSize - 1) / 2)
    paddingImage = np.pad(kernel, (padding, padding))
    columnEnd = paddingImage.shape[1] - 1
    print(paddingImage)
    for columnIndex in range(padding, columnEnd - padding):
        pass


if __name__ == "__main__":
    img = Image.open("./img/img_1.png")
    kSize = 3
    kernel1 = sobel_x()
    img_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    convolutionFig(img_arr, kernel1)
