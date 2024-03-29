import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = ' '.join(sentences).split()
word_list = list(set(word_list))
word_dict = {w : i for i, w in enumerate(word_list)}
number_dict = {i : w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# NNLM parameter
n_step = 2
n_hidden = 2
m = 2

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        sen = sen.split()
        input = [word_dict[i] for i in sen[:-1]]
        target = word_dict[sen[-1]]
        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step *m , n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m)
        tanh = torch.tanh(self.d + torch.mm(X, self.H))
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U)
        return output


model = NNLM()
input_batch, target_batch = make_batch(sentences)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))


for epoch in range(5000):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

predict = model(input_batch).data.max(1, keepdim= True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])


        


for sen in sentences:
    sen = sen.split()
    input = torch.LongTensor([word_dict[n] for n in sen[:-1]])

