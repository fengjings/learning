# -*- coding: utf-8 -*-

import torch
a = torch.randn(2,3)
print(a)

b = torch.pow(a, 2)
print(b)

c = torch.pow(a, a)
print(c)

########################### NN
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = torch.randn(batch_n, input_data)
y = torch.randn(batch_n, output_data)

w1 = torch.randn(input_data, hidden_layer)
w2 = torch.randn(hidden_layer, output_data)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    h1 = x.mm(w1) #100*1000
    h1 = h1.clamp(min = 0)
    y_pred = h1.mm(w2) #100*10
    
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch,loss))
    
    grad_y_pred = 2*(y_pred - y)
    grad_w2 = h1.t().mm(grad_y_pred)
    
    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    
    grad_w1 = x.t().mm(grad_h)
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2
    
    
#####################################autograde
print('-'*30)
import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)
# x = torch.randn((batch_n, input_data), requires_grad = False)
# y = torch.randn((batch_n, output_data), requires_grad = False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)
#w1 = torch.randn((input_data, hidden_layer), requires_grad = True)
#w2 = torch.randn((hidden_layer, output_data), requires_grad = True)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch,loss.data))
    
    loss.backward()
    
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data
    
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    
    
    

##################################### define model
print('-'*30)    
import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, a, w1, w2):
        x = torch.mm(a, w1)
        x = torch.clamp(x, min = 0) #same as relu
        x =torch.mm(x, w2)
        return x
    def backward(self):
        pass

model = Model()

x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)

epoch_n = 30
learning_rate = 1e-6

for epoch in range(epoch_n):
    y_pred = model(x, w1, w2)
    
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch,loss.data))
    
    loss.backward()
    
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    
    w1.grad.data.zero_()
    w2.grad.data.zero_()

#####################################torch.nn
print('-'*30)   
###############################################
import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

models = torch.nn.Sequential(
  torch.nn.Linear(input_data, hidden_layer),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden_layer, output_data)
)
print(models)

from collections import OrderedDict
models2 = torch.nn.Sequential(OrderedDict([
("Line1",torch.nn.Linear(input_data, hidden_layer)),
("Relu1",torch.nn.ReLU()),
("Line2",torch.nn.Linear(hidden_layer, output_data))])
)
print(models2)

import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

models = torch.nn.Sequential(
torch.nn.Linear(input_data, hidden_layer),
torch.nn.ReLU(),
torch.nn.Linear(hidden_layer, output_data)
)

epoch_n = 10000
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()

for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    if epoch%1000 == 0:
        print("Epoch:{}, Loss:{:.4f}".format(epoch,loss.data))
    models.zero_grad()
    loss.backward()
    for param in models.parameters():
        param.data -= param.grad.data*learning_rate


#####################################torch.optim
print('-'*30)   
###############################################
import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad=False)
#x = torch.randn((batch_n, input_data), requires_grad = False)
#y = torch.randn((batch_n, output_data), requires_grad=False)

models = torch.nn.Sequential(
torch.nn.Linear(input_data, hidden_layer),
torch.nn.ReLU(),
torch.nn.Linear(hidden_layer, output_data)
)

epoch_n = 20
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()

optimzer = torch.optim.Adam(models.parameters(), lr = learning_rate)

for epoch in range(20):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    print("Epoch:{}, Loss:{:.4f}".format(epoch,loss.data))
    optimzer.zero_grad()
    
    loss.backward()
    optimzer.step()




























