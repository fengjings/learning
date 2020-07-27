import torch
import platform

print("PyTorch version:{}".format(torch.__version__))
print("Python version:{}".format(platform.python_version()))

import time
import torch
# use CPU
for i in range(1,5):
    start = time.time()
    a = torch.FloatTensor(i*100,1000,1000)
    a = torch.matmul(a,a)
    end = time.time()-start
    print(end)
print('---')

## use GPU
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.current_device())

import os
import numpy as np
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
CUDA_VISIBLE_DEVICES=str(np.argmax(memory_gpu))
print(CUDA_VISIBLE_DEVICES)
os.system('rm tmp')


print(memory_gpu)
for i in range(1,10):
    start = time.time()
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    a = torch.FloatTensor(i*100,1000,1000)
    a = a.to(device)
    a = torch.matmul(a,a)
    end = time.time()-start
    print(end)