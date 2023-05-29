import torch
import torch.nn as nn
import torch.nn.functional as F
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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
batch_size = 32

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))   
        x = torch.sigmoid(self.fc2(x))
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
        output = model(data).reshape(-1)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    model.eval()
    test_loss = []
    with torch.no_grad():
        for data, target in test_loader :
            output = model(data).reshape(-1)
            test_loss.append(criterion(output, target.float()).item()) # sum up batch loss
    los.append(np.mean(test_loss))

L = len(los)
x = np.array(range(L))   
plt.plot(x, los) 
plt.show()   
torch.save(model, "./data/model.pth")