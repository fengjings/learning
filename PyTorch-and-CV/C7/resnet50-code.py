import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

data_dir = "/home2/yks/program/data/DogsVSCats"

data_transform = {x:transforms.Compose([#transforms.Scale([224,224]),
                                        transforms.Resize([224,224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
                  for x in ["train", "valid"]}

image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                         transform = data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x:torch.utils.data.DataLoader(dataset= image_datasets[x],
                                            batch_size = 16,
                                            shuffle = True)
              for x in ["train", "valid"]}

model = models.resnet50(pretrained=True)
print(model)


Use_gpu = torch.cuda.is_available()
print(Use_gpu)
print('current_device',torch.cuda.current_device())
torch.cuda.set_device(3)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print(device)
print('current_device',torch.cuda.current_device())
print("-"*10)


for parma in model.parameters():
    parma.requires_grad = False

model.fc = torch.nn.Linear(2048, 2)
print(model)

    
if Use_gpu:
    model = model.cuda()
model.to(device)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters())

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr = 0.00001)
epoch_n = 5


time_open = time.time()
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n - 1))
    print("-"*10)
    
    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)
            
        running_loss = 0.0
        running_corrects = 0
        
        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            
            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)
            X = X.to(device)
            y = y.to(device)
                
                
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, y)
            if phase == "train":
                loss.backward()
                optimizer.step()
            
            running_loss += loss.data
            running_corrects += torch.sum(pred == y.data)
            
            if batch%5 == 0 and phase == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(batch, running_loss/batch,100*running_corrects/(16*batch)))
            
        epoch_loss = running_loss*16/len(image_datasets[phase])
        epoch_acc = 100*running_corrects/len(image_datasets[phase])
        
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss,epoch_acc))
            
time_end = time.time() - time_open
print(time_end)













