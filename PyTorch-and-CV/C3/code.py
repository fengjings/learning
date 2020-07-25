
import torch
import platform
print("PyTorch version:{}".format(torch.__version__))
print("Python version:{}".format(platform.python_version()))

import time
import torch
# use CPU
for i in range(1,20):
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
for i in range(1,20):
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.FloatTensor(i*100,1000,1000)
    a = a.to(device)
    a = torch.matmul(a,a)
    end = time.time()-start
    print(end)