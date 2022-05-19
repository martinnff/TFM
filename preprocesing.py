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
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.skopt import SkOptSearch

n_folds = 10

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

for i in range(n_folds):
    i+=1
    datos1 = split_sp(path='./datos2/grafos/train/sp1',seed=i,parcela=0)
    parcela=(len(datos1[0])+len(datos1[1]))
    datos2 = split_sp(path='./datos2/grafos/train/sp2',seed=i,parcela=parcela)
    parcela=parcela+(len(datos2[0])+len(datos2[1]))
    datos3 = split_sp(path='./datos2/grafos/train/sp3',seed=i,parcela=parcela)


    test = []
    test.extend(datos1[0])
    test.extend(datos2[0])
    test.extend(datos3[0])

    train = []
    train.extend(datos1[1])
    train.extend(datos2[1])
    train.extend(datos3[1])

    with open(f"./datos2/train_graphs{i}_90.pkl",'wb') as f:
        pickle.dump(train,f)
    with open(f"./datos2/test_graphs{i}_90.pkl",'wb') as f:
        pickle.dump(test,f)
