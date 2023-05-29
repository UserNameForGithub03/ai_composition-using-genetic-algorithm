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

x_good = np.load('./data/good1.npy')
x_bad = np.load('./data/bad.npy')
x_train_good = x_good [ : 300]
x_test_good = x_good [300 : ] 
x_train_bad = x_bad [ : 450]
x_test_bad = x_bad [450 : ]

y_train_good = np.ones(x_train_good.shape[0]) 
y_train_bad = np.zeros(x_train_bad.shape[0])
y_test_good = np.ones(x_test_good.shape[0]) 
y_test_bad = np.zeros(x_test_bad.shape[0])
batch_size = 32

x_train = np.concatenate((x_train_good, x_train_bad))
y_train = np.concatenate((y_train_good, y_train_bad))
x_test = np.concatenate((x_test_good, x_test_bad))
y_test = np.concatenate((y_test_good, y_test_bad))

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
batch_size = 32

device = torch.device('cpu')
seq_len = 32
lstm_layers = 1
bidirectional_lstm = False
D = 1 
dim = 60
BASE = 40 
hidden_size = 16
prev_h = np.random.random([D * lstm_layers, batch_size, hidden_size]).astype(np.float32)
prev_c = np.random.random([D * lstm_layers, batch_size, hidden_size]).astype(np.float32)
prev_h = torch.FloatTensor(prev_h).to(device)
prev_c = torch.FloatTensor(prev_c).to(device)
np.save("./data/prev_c", prev_c)
np.save("./data/prev_h", prev_h)

class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        data = self.X[index].astype('float32')
        label = self.y[index].astype('int64') 
        return data, label
    def __len__(self):
        return len(self.X)

train_dataset = Data(X = x_train, y = y_train)
test_dataset = Data(X = x_test, y = y_test)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

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
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size = dim, hidden_size = hidden_size, batch_first = True, bidirectional = bidirectional_lstm)
        self.linear1 = nn.Linear(in_features = hidden_size, out_features = 16)
        self.linear2 = nn.Linear(in_features = 16, out_features = 1)
    def forward(self, x):
        x, _ = self.lstm(x, (prev_h, prev_c))
        x = torch.mean(x, dim = 0)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

model = Net()
model.to(device='cpu')
criterion = F.binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
EPOCHS = 100
loss_train = []
los = []
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x = Embedding (data)
        x = torch.tensor(x).float()
        output = model(x).reshape(-1)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    model.eval()
    test_loss = []
    with torch.no_grad():
        for data, target in test_loader :
            x = Embedding (data)
            x = torch.tensor(x).float()
            output = model(x).reshape(-1)
            test_loss.append(criterion(output, target.float()).item()) # sum up batch loss
    los.append(np.mean(test_loss))

L = len(los)
x = np.array(range(L))   
plt.plot(x, los) 
plt.show()   
torch.save(model, "./data/rnn.pth")