vgg_source = [21, -2]#取出conv4-3进行回归预测，分类预测
for k, v in enumerate(vgg_source):
    print(k)
    print(v)

import torch
#print(torch.version.cuda)



import torch
 
if __name__ == '__main__':
    print("torch version: ?", torch.__version__)
    print("Support CUDA ?: ", torch.cuda.is_available())
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print(xx)

    y = torch.randn(2, 3)
    yy = y.cuda()
    print(yy)

    zz = xx + yy
    print(zz)

    # CUDNN TEST
    from torch.backends import cudnn
    print("Support cudnn ?: ",cudnn.is_acceptable(xx))