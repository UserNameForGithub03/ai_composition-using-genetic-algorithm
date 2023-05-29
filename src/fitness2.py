import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, Module
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold = np.inf)

device = torch.device('cpu')
seq_len = 32
lstm_layers = 1
bidirectional_lstm = False
D = 1 
dim = 60
BASE = 40 
hidden_size = 16
prev_h = np.load("./data/prev_h.npy")
prev_c = np.load("./data/prev_c.npy")
prev_h = torch.FloatTensor(prev_h).to(device)
prev_c = torch.FloatTensor(prev_c).to(device)

def Embedding(x_batch) : 
    res = np.empty([seq_len, x_batch.shape[0], dim]) 
    for id, x in enumerate(x_batch) :
        for i in range(0, seq_len) : 
            y = np.zeros(dim) 
            if(x[i * 2] >= BASE) : 
                y[int (x[i * 2] - BASE)] = 1.0 
            y[dim - 1] = x[i * 2 + 1]
            res[i][id] = y 
    return res 
    
class Net(Module):
    def __init__(self) :
        super(Net, self).__init__()
        self.lstm = LSTM(input_size = dim, hidden_size = hidden_size, batch_first = True, bidirectional = bidirectional_lstm)
        self.linear1 = nn.Linear(in_features = hidden_size, out_features = 16)
        self.linear2 = nn.Linear(in_features = 16, out_features = 1)
    def forward(self, x) :
        x, _ = self.lstm(x, (prev_h, prev_c))
        x = torch.mean(x, dim = 0)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

model = torch.load("./data/rnn.pth")
model.to(device='cpu')

def fitness (x) : 
    x = Embedding (x.reshape(1, -1))
    x = torch.tensor(x).float()
    return model(x).item()