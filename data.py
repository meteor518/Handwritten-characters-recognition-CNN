# coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe
"""

import os
from PIL import Image
import numpy as np


# 读取文件夹mnist下图片，图片为灰度图，图像大小28*28
def load_data():
    data = np.empty((42000, 28, 28, 1), dtype="float32")
    label = np.empty((42000,), dtype="uint8")

    imgs = os.listdir("./mnist")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./mnist/" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
        label[i] = int(imgs[i].split('.')[0])
    return data, label
