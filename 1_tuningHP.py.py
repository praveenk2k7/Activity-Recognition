# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:37:10 2019

@author: Praveen
"""
import pandas as pd
import numpy as np

data=pd.read_csv('Data.csv')

df=data.drop(['attitude_sum_roll','attitude_sum_pitch','attitude_sum_yaw','gravity_sum_x','gravity_sum_y', 'gravity_sum_z', 'rotationRate_sum_x','rotationRate_sum_y', 'rotationRate_sum_z', 'userAcceleration_sum_x',
       'userAcceleration_sum_y','attitude_sumSS_roll', 'attitude_sumSS_pitch','attitude_sumSS_yaw', 'gravity_sumSS_x', 'gravity_sumSS_y',
       'gravity_sumSS_z', 'rotationRate_sumSS_x', 'rotationRate_sumSS_y','rotationRate_sumSS_z', 'userAcceleration_sumSS_x',
       'userAcceleration_sumSS_y', 'userAcceleration_sumSS_z','Activities_Types'],inplace=False,axis=1)


from sklearn.model_selection import train_test_split
import torch


labels=data['Activities_Types'].values
labels=labels-1
x_train,x_test,y_train,y_test=train_test_split(df,labels,test_size=0.2)    

x_train = torch.tensor(x_train.values,dtype=torch.float)
x_test = torch.tensor(x_test.values,dtype=torch.float)

y_train = torch.tensor(y_train,dtype=torch.long)
y_test = torch.tensor(y_test,dtype=torch.long)

y = np.zeros((y_train.shape[0], 6))
y[np.arange(y_train.shape[0]), y_train-1] = 1
y=torch.tensor(y,dtype=torch.long)
y1 = np.zeros((y_test.shape[0], 6))
y1[np.arange(y_test.shape[0]), y_test-1] = 1
y1=torch.tensor(y1,dtype=torch.long)


#creating the dataset class


import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid


class act(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X.float()
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        else:     
            return self.X[index]
        

train_dataset=act(X=x_train,y=y_train)
valid_dataset=act(X=x_test,y=y_test)


import torch.nn as nn

class Net(nn.Module):
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(Layers[0],Layers[1]))
        Layers=Layers[1:]
        for i in range(n_layers):
            for input_size, output_size in zip(Layers, Layers[1:-1]):
                self.hidden.append(nn.Linear(input_size, output_size))
        self.hidden.append(nn.Linear(Layers[-2],Layers[-1]))
    
    
        
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

#model=MLP()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#loss_fn = nn.CrossEntropyLoss()



# Define the range

num_layers=[1]#,3,4,5]
num_neurons=[5,10]#,15,20,25]
epochs=[25]#,100,125]
l_r=[0.001]#,0.01,0.1,1]
batch_size=[32]#,64,128,256]

import itertools 
comb=list(itertools.product(num_layers,num_neurons,epochs,l_r,batch_size))


tot_acc=[]
valid_acc=[]
max_acc=0
train_loss=[]
val_loss=[]
for s in range(len(comb)):
    n_layers,n_neurons,epochs,lr,b_size=comb[s]
    in_size=45
    out_size=6

    Layers = [in_size,n_neurons,n_neurons,out_size]
    model = Net(Layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True)
    valid_loader=DataLoader(dataset=valid_dataset, batch_size=b_size, shuffle=True)




    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
   


    for epoch in range(epochs):
        model.train()
    
        train_losses = []
        valid_losses = []
        for i, (images, labels) in enumerate(train_loader):
        
        
        
            outputs = model(images)#forward prop
            loss = loss_fn(outputs,labels)#calculate loss
            optimizer.zero_grad()#zero the grad
            loss.backward()#backward pass
            optimizer.step()#update the parameters
        
            train_losses.append(loss.item())
        
        ##   print(f'{i * 128} / 50000')
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(valid_loader):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            
                    valid_losses.append(loss.item())
            
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            mean_train_losses.append(np.mean(train_losses))
            mean_valid_losses.append(np.mean(valid_losses))
    
            accuracy = 100*correct/total
            valid_acc_list.append(accuracy)
            #print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%'\
            # .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))
        
    tot_acc.append(accuracy)
    if max_acc<np.max(tot_acc):
        max_acc=np.max(tot_acc)
        valid_acc=valid_acc_list
        train_loss=mean_train_losses
        val_loss=mean_valid_losses
    print("set: {}, accuracy: {}".format(s,accuracy))
print("best hyper parameters: n_layers: {},n_neurons: {},epochs: {},lr: {},b_size: {}, with accuracy: {}".format(comb[np.argmax(tot_acc)][0],comb[np.argmax(tot_acc)][1],comb[np.argmax(tot_acc)][2],comb[np.argmax(tot_acc)][3],comb[np.argmax(tot_acc)][4],max_acc))



