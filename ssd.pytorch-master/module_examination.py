from __future__ import print_function
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd

print('import module pass')

print('torch.cuda.is_available ', torch.cuda.is_available())
print('torch.cuda.device_count ', torch.cuda.device_count())
print('torch.cuda.get_device_name ', torch.cuda.get_device_name())
print('torch.cuda.current_device ', torch.cuda.current_device())
torch.cuda.set_device()
print('torch.cuda.current_device', torch.cuda.current_device())