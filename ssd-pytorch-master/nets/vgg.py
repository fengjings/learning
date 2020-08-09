import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

def vgg(i):
    layers = []
    in_channels = i # 通道数，3
    for v in base:
        if v == 'M':# 池化操作
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':# 池化操作 把不足的边保留下来，计算方式类似于在后面补0后进行池化
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v #进行卷积的通道数，第一次为3，之后为base中的数值
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# 300,300,3

# 64, 64, 'M'：
# Conv1_1 300,300,64
# Conv1_2 300,300,64
# Pooling1 150,150,64
# 128, 128, 'M'：
# Conv2_1 150,150,128
# Conv2_2 150,150,128
# pooling2 75,75,128
# 256, 256, 256, 'C'：
# Conv3_1 75,75,256
# Conv3_2 75,75,256
# Conv3_3 75,75,256
# pooling3 38,38,256
# 512, 512, 512, 'M'：
# Conv4_1 38,38,512
# Conv4_2 38,38,512
# Conv4_3 38,38,512
# pooling4 19,19,512
# 512, 512, 512：
# Conv5_1 19,19,512
# Conv5_2 19,19,512
# Conv5_3 19,19,512

# pool5 19*19*512
# conv6 19*19*1024
# conv7 19*19*1024