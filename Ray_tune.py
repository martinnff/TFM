from functools import partial
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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def split_sp(path,seed=1,split=0.9,parcela=0):
    np.random.seed(seed)
    random.seed(seed)
    parcelas = os.listdir(path)
    n_sample = round(len(parcelas)*split)
    train_ind = np.random.choice(np.linspace(1,len(parcelas),len(parcelas)),n_sample,replace=False).astype(int)
    test_ind = np.delete(np.linspace(1,len(parcelas),len(parcelas)),train_ind-1).astype(int)
    test_graphs = []
    train_graphs = []

    parcela = parcela
    for itrain in train_ind:
        filename = f"{path}/parcela{itrain}"
        archivos = os.listdir(filename)
        ngrafos=int(len(archivos)/4)
        parcela+=1
        for i in range(ngrafos):
            i+=1
            edges = pd.read_csv(f"{filename}/el{i}.csv").iloc[:,1:]
            attributes = pd.read_csv(f"{filename}/z{i}.csv").iloc[:,1]
            label = pd.read_csv(f"{filename}/y{i}.csv").iloc[:,1]
            weights = pd.read_csv(f"{filename}/ea{i}.csv").iloc[:,1]
            weights = torch.tensor(weights.to_numpy(),dtype=torch.float)
            edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            edge_idx -= 1
            attrs = torch.tensor(attributes.to_numpy(), dtype=torch.float)
            np_lab = label.to_numpy()
            y = torch.tensor(np_lab, dtype=torch.long)-1
            y = y[0]
            graph = Data(x=attrs, edge_index=edge_idx,  y=y, edge_attr = weights,parcela=parcela)
            train_graphs.append(graph)
    for itest in test_ind:
        filename = f"{path}/parcela{itest}"
        archivos = os.listdir(filename)
        ngrafos=int(len(archivos)/4)
        parcela+=1
        for i in range(ngrafos):
            i+=1
            edges = pd.read_csv(f"{filename}/el{i}.csv").iloc[:,1:]
            attributes = pd.read_csv(f"{filename}/z{i}.csv").iloc[:,1]
            label = pd.read_csv(f"{filename}/y{i}.csv").iloc[:,1]
            weights = pd.read_csv(f"{filename}/ea{i}.csv").iloc[:,1]
            weights = torch.tensor(weights.to_numpy(),dtype=torch.float)
            edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            edge_idx -= 1
            attrs = torch.tensor(attributes.to_numpy(), dtype=torch.float)
            np_lab = label.to_numpy()
            y = torch.tensor(np_lab, dtype=torch.long)-1
            y = y[0]
            graph = Data(x=attrs, edge_index=edge_idx,  y=y, edge_attr = weights,parcela=parcela)
            test_graphs.append(graph)
    return [test_graphs,train_graphs]


#datos1 = split_sp(path='./datos2/grafos/train/sp1',parcela=0)
#parcela=(len(datos1[0])+len(datos1[1]))
#datos2 = split_sp(path='./datos2/grafos/train/sp2',parcela=parcela)
#parcela=parcela+(len(datos2[0])+len(datos2[1]))
#datos3 = split_sp(path='./datos2/grafos/train/sp3',parcela=parcela)


# test = []
# test.extend(datos1[0])
# test.extend(datos2[0])
# test.extend(datos3[0])
#
# train = []
# train.extend(datos1[1])
# train.extend(datos2[1])
# train.extend(datos3[1])
#
# with open("./datos2/train_graphs1_90.pkl",'wb') as f:
#     pickle.dump(train,f)
# with open("./datos2/test_graphs1_90.pkl",'wb') as f:
#     pickle.dump(test,f)


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


# train_dataset = GraphDataset("./datos/train_graphs1_90.pkl")
# test_dataset = GraphDataset("./datos/test_graphs1_90.pkl")
#


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
    def __init__(self, hid = 64,
                 in_head = 16,
                 out_features = 4,
                 s_fc1 = 2048,
                 s_fc2 = 1024):
        super(GAT, self).__init__()

        self.hid = hid
        self.in_head = in_head
        self.in_features = 1
        self.out_features = out_features
        self.s_fc1 = s_fc1
        self.s_fc2 = s_fc2

        self.conv1 =  GATv2Conv(self.in_features, self.out_features,edge_dim=1,heads=self.in_head,concat=True)
        self.conv2 =  SAGEConv(self.out_features*self.in_head, self.hid,normalize=False)
        self.norm1=GraphNorm(self.out_features*self.in_head)
        self.fc1 = nn.Linear(self.hid*2,self.s_fc1)
        self.fc2 = nn.Linear(self.s_fc1,self.s_fc2)
        self.fc3 = nn.Linear(self.s_fc2,3)
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
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
        x = F.dropout(x, p=dp, training=self.training)

        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=dp, training=self.training)

        x1 = global_max_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        x = torch.cat((x1,x2),1)

        x = self.fc1(x)
        x = F.dropout(x, p=dp, training=self.training)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.dropout(x, p=dp, training=self.training)
        x = F.relu(x)

        x = self.fc3(x)

        return x





def train_graphs(config):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    model = GAT(config['hid'],
            config['in_head'],
            config['out_features'],
            config['s_fc1'],
            config['s_fc2'])

    model.to(device)
    print(model.train())


    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    criterion = torch.nn.CrossEntropyLoss(torch.tensor([1.1,1.05,1.0]).to(device))


    trainset = GraphDataset("/home/martin/Master/TFM/grafitos/datos/train_graphs1_90.pkl")
    testset = GraphDataset("/home/martin/Master/TFM/grafitos/datos/test_graphs1_90.pkl")

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True)

    valloader = DataLoader(
        testset,
        batch_size=int(200),
        shuffle=True,
        drop_last=True)

    model.train()
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        lss=0
        count=0
        for i, data in enumerate(trainloader):

            batch=data.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            running_loss+=loss.item()
            loss.backward()
            if(i!=0 and i%5==0):
                torch.nn.utils.clip_grad_norm_(model.parameters(),2)
                optimizer.step()
                optimizer.zero_grad()

            count+=1


        running_loss/=count
        epoch_steps += 1


        print('[Epoch %4d/%4d] Loss: % 2.2e' % (epoch + 1, 50, running_loss))

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader):
            with torch.no_grad():
                model.eval()
                data = data.to(device)

                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()

                loss = criterion(outputs, data.y)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        print('[Epoch %4d/%4d] Test Loss: % 2.2e' % (epoch + 1, 50, val_loss / val_steps))
        print('[Epoch %4d/%4d] Test Accuracy: % 2.2e' % (epoch + 1, 50, correct / total))

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")



config = {
    "hid": tune.choice([4, 8, 16,32, 64,88,128]),
    "in_head": tune.choice([2, 4, 8, 16,20, 24]),
    "out_features": tune.choice([2, 4, 6, 8,12,16]),
    "s_fc1": tune.choice([256, 512, 1024, 2048,3096]),
    "s_fc2": tune.choice([256, 512, 1024, 2048,3096]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "wd": tune.loguniform(5e-4, 1e-6),
    "batch_size": tune.uniform(20,1000)
}

gpus_per_trial = 0
result = tune.run(train_graphs,
    resources_per_trial={"cpu":4, "gpu": gpus_per_trial},
    config=config,
    num_samples=4,
    scheduler=ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=2,
        grace_period=1,
        reduction_factor=2),
    progress_reporter=CLIReporter(
        parameter_columns=["hid","in_head","out_features","s_fc1","s_fc2","lr","wd"],
        metric_columns=["loss", "accuracy", "training_iteration"]))


analysis = tune.ExperimentAnalysis(

    experiment_checkpoint_path="/home/martin/ray_results/train_graphs_2022-05-12_08-46-25/experiment_state-2022-05-12_08-46-26.json")

# Get a dataframe for the last reported results of all of the trials
df = analysis.results_df
print(df)
# Get a dataframe for the max accuracy seen for each trial
df = analysis.dataframe(metric="loss", mode="min")

# Get a dict mapping {trial logdir -> dataframes} for all trials in the experiment.
all_dataframes = analysis.trial_dataframes
print(all_dataframes)
# Get a list of trials
trials = analysis.trials
print(trials)
