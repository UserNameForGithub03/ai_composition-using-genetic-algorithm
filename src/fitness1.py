import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


import numpy as np

from matplotlib import pyplot as plt

np.set_printoptions(threshold = np.inf)

scaler = StandardScaler()
x_good = np.load('./data/good1.npy')
x_bad = np.load('./data/bad.npy')
data = np.concatenate((x_good, x_bad))
scaler = StandardScaler()
data = scaler.fit_transform(data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))   
        x = torch.sigmoid(self.fc2(x))
        return x
model = torch.load("./data/model.pth")
model.to(device='cpu')

# test = np.load("./data/bad.npy")

def fitness (x) : 
    x = scaler.transform(x.reshape(1, -1))
    return model(torch.from_numpy(x).float()).item()

# y = fitness(test[1])
