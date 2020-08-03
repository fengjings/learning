import torch
import platform
print("PyTorch version:{}".format(torch.__version__))
print("Python version:{}".format(platform.python_version()))

import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
use_gpu = torch.cuda.is_available()


def loadimg(path = None):
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img


transform = transforms.Compose([#transforms.Scale([224,224]),
                                transforms.Resize([224,224]),
                                transforms.ToTensor()])

content_img = loadimg("images/cat.jpg")
content_img = Variable(content_img).cuda()
content_img = content_img.to(device)

style_img = loadimg("images/style.jpg")
style_img = Variable(style_img).cuda()
style_img = style_img.to(device)


class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach()*weight
        self.loss_fn = torch.nn.MSELoss()
        
    def forward(self, input):
        self.loss = self.loss_fn(input*self.weight, self.target)
        return input
    
    def backward(self):
        self.loss.backward(retain_graph = True)
        return self.loss


class Gram_matrix(torch.nn.Module):
    
    def forward(self, input):
        a,b,c,d = input.size()
        feature = input.view(a*b, c*d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a*b*c*d)

class Style_loss(torch.nn.Module):
    
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach()*weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = Gram_matrix()
        
    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)
        self.loss = self.loss_fn(self.Gram, self.target)
        return input
    
    def backward(self):
        self.loss.backward(retain_graph = True)
        return self.loss



cnn = models.vgg16(pretrained=True).features

if use_gpu:
    cnn = cnn.cuda()
    model = copy.deepcopy(cnn)
        
cnn = cnn.to(device)
model = copy.deepcopy(cnn) 

content_layer = ["Conv_3"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]
content_losses = []
style_losses = []
conten_weight = 1
style_weight = 1000


new_model = torch.nn.Sequential()
model = copy.deepcopy(cnn)
gram = Gram_matrix()

if use_gpu:
    new_model = new_model.cuda()
    gram = gram.cuda()
    
new_model = new_model.to(device)
gram = gram.to(device)
    
index = 1
for layer in list(model)[:8]:
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_"+str(index)
        new_model.add_module(name, layer)

        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = Content_loss(conten_weight, target)
            new_model.add_module("content_loss_"+str(index), content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module("style_loss_"+str(index), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = "Relu_"+str(index)
        new_model.add_module(name, layer)
        index = index+1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = "MaxPool_"+str(index)
        new_model.add_module(name, layer)

input_img = content_img.clone()
parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])
epoch_n = 300
epoch = [0]

while epoch[0] <= epoch_n:
        def closure():
            optimizer.zero_grad()
            style_score = 0
            content_score = 0
            parameter.data.clamp_(0,1)
            new_model(parameter)

            for sl in style_losses:
                style_score += sl.backward()
            
            for cl in content_losses:
                content_score += cl.backward()

            epoch[0] += 1

            if epoch[0] % 50 == 0:
                print('Epoch:{} Style Loss: {:4f} Content Loss:{:4f}'.format(epoch[0],style_score, content_score))

            return style_score+content_score

        optimizer.step(closure)



