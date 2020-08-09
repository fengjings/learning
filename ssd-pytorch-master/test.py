vgg_source = [21, -2]#取出conv4-3进行回归预测，分类预测
for k, v in enumerate(vgg_source):
    print(k)
    print(v)

import torch
#print(torch.version.cuda)