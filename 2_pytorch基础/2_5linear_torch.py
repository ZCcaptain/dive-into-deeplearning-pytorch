import torch 
from IPython import display
from matplotlib import pyplot as plt 
import numpy as np 
import random
import torch.utils.data as Data
import torch.nn as nn 
#1.构建数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] +true_b
labels += torch.tensor(np.random.normal(0, 0.01,size=labels.size()), dtype=torch.float)

batch_size = 8

#2.读取数据
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

#3.定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
net = LinearNet(2)
print(net.linear.weight)

net = nn.Sequential(
nn.Linear(num_inputs, 1)

)
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

from collections import OrderedDict
net = nn.Sequential(OrderedDict([
('linear', nn.Linear(num_inputs, 1))
]))
# print(net)
# print(net[0])

#4.初始化模型参数
params = net.parameters()
# for param in params:
#     print(param)

from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
params = net.parameters()
for param in params:
    print(param)

loss = nn.MSELoss()

import torch.optim as optim 
optimizer = optim.SGD(net.parameters(), lr = 0.03)

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) **2 
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for x, y in data_iter:
        optimizer.zero_grad()
        output = net(x)
        l = loss(output, y.view(-1, 1))
        l2 = squared_loss(output, y).mean()
        print(l, l2)
        break
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)