
from os.path import exists
import random
import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import metrics
import scipy.sparse as scpy
from torch.nn import Linear
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.data as data
import torch_geometric.data.data as Batch
import torch.utils.data as data_utils
import torch_geometric.transforms as T
from torch.nn.init import xavier_uniform
from sklearn.metrics import roc_curve, auc
import torch_geometric.datasets as datasets
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch_geometric.transforms as transforms
from sklearn.preprocessing import label_binarize
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import InMemoryDataset, Data
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch_geometric.nn import  GATv2Conv, GraphNorm,  SAGEConv, global_mean_pool, global_max_pool


class GraphDataset(InMemoryDataset):
    def __init__(self, path):
        self.path = path
        self.data_list = []
        with open(path,'rb') as f:
            self.data_list.extend(pickle.load(f))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


train_dataset = GraphDataset("./datos2/train_graphs1_90.pkl")
test_dataset = GraphDataset("./datos2/test_graphs1_90.pkl")



# funcion para inicializar pesos de la red
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

dp=0.2




class GAT(torch.nn.Module):
    def __init__(self, hid = 1,
                 in_head = 4,
                 out_features = 1,
                 s_fc1 = 4,
                 s_fc2 = 2,
                 cv2 = 0,
                 dp = 0.1):
        super(GAT, self).__init__()

        self.hid = int(64*hid)
        self.in_head = int(4*in_head)
        self.in_features = 1
        self.out_features = int(4*out_features)
        self.s_fc1 = int(512*s_fc1)
        self.s_fc2 = int(512*s_fc2)
        self.cv2 = cv2
        self.dp = dp

        self.conv1 =  GATv2Conv(self.in_features, self.out_features,edge_dim=1,heads=self.in_head,concat=True)
        if(self.cv2==1):
            self.conv2 =  GATv2Conv(self.out_features*self.in_head, self.out_features*self.in_head,edge_dim=1,heads=self.in_head,concat=False)
            self.conv2.apply(init_weights)
        self.conv3 =  SAGEConv(self.out_features*self.in_head, self.hid,normalize=False)
        self.norm1=GraphNorm(self.out_features*self.in_head)
        self.fc1 = nn.Linear(self.hid*2,self.s_fc1)
        self.fc2 = nn.Linear(self.s_fc1,self.s_fc2)
        self.fc3 = nn.Linear(self.s_fc2,3)
        self.conv1.apply(init_weights)
        self.conv3.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = torch.unsqueeze(edge_attr,1)
        x = torch.unsqueeze(x,-1)

        x = self.conv1(x,edge_index,edge_attr)
        x = F.relu(x)
        x = self.norm1(x,batch)
        x = F.dropout(x, p=self.dp, training=self.training)

        if(self.cv2==1):
            x = self.conv2(x,edge_index,edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dp, training=self.training)



        x = self.conv3(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp, training=self.training)

        x1 = global_max_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        x = torch.cat((x1,x2),1)

        x = self.fc1(x)
        x = F.dropout(x, p=self.dp, training=self.training)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.dropout(x, p=self.dp, training=self.training)
        x = F.relu(x)

        x = self.fc3(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config={'hid': 5.835391375421207,
 'in_head': 5.33870021958179,
 'out_features': 5.219630746587441,
 's_fc1': 7.0987821602409324,
 's_fc2': 3.6571746453029723,
 'lr': 0.00012180182994336132,
 'wd': 3.668794039498245e-07,
 'dp': 0.18423803058335414,
 'batch_size': 100,
 'hold_gradient': 1.3657816278195805,
 'cv2': 0,
 'epoch': 110.68922901985783}
print(device)

model = GAT(config['hid'],
                config['in_head'],
                config['out_features'],
                config['s_fc1'],
                config['s_fc2'],
                config['cv2'],
                config['dp']).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

data_test=DataLoader(test_dataset,batch_size=50,shuffle=True,drop_last=True)

for i , batch in enumerate(data_test):
    dat = batch.to(device)
    batch = data.Batch(batch)

tbn=0
optimizer.zero_grad()
l=torch.nn.CrossEntropyLoss(torch.tensor([1.1,1.05,1.0]).to(device))
print(model.train())
for epoch in range(int(config['epoch'])):
    model.train()
    lss=0
    train=DataLoader(train_dataset,batch_size=int(config['batch_size']),shuffle=True,drop_last=True)
    count=0
    for idx, batch in enumerate(train):
        batch=batch.to(device)
        out = model(batch)
        loss = l(out, batch.y)
        lss+=loss.item()
        loss.backward()
        if(idx!=0 and idx%int(config['hold_gradient'])==0):
            torch.nn.utils.clip_grad_norm_(model.parameters(),2)
            optimizer.step()
            optimizer.zero_grad()
        count+=1


    tbn+=1
    lss=lss/count
    writer.add_scalar('Training/Loss', float(lss), tbn)
    print('[Epoch %4d/%4d] Loss: % 2.2e' % (epoch + 1, 250, lss))
    model.eval()
    loss2=0
    count2=0
    for i, batch in enumerate(data_test):
        count2+=1
        dat = batch.to(device)
        with torch.no_grad():
            pred=model(dat)
            loss2 += l(pred, dat.y)
    loss2 = loss2/count2
    writer.add_scalar('test/Loss', float(loss2), tbn)
    print('[Epoch %4d/%4d] Test Loss: % 2.2e' % (epoch + 1, 250, loss2))
    if(loss2<0.76):
        break

writer.flush()
writer.close()
