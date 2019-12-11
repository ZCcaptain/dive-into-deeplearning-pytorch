import torch
import random
import zipfile
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
#1.读取数据集
with zipfile.ZipFile('/home/zhaochao/workspace/dive-into-deeplearning-pytorch/6.循环神经网络/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]

#2.建立字符索引
idx_to_char = list(set(corpus_chars))
char_to_idx = dict((char, i) for i, char in enumerate(idx_to_char))
vocab_size = len(char_to_idx)

corpus_indices = [char_to_idx[ch] for ch in corpus_chars]
#3.one-hot向量
def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    #scatter 三个参数 维度（就是对第几维操作，其他维不动）， 索引， iput
    #在第一维度也就是n_class维度对索引的第一维度指出的位置 改为input
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


#4.初始化模型参数

num_inputs, num_hiddens, num_outputs = vocab_size, 256,  vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('will use', device)

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32, requires_grad=True))
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

#5.定义模型
#5.1 定义隐状态初始化函数
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device, dtype=torch.float32), )

#5.2 定义一次n_step个时间步里的隐藏状态和输出的计算
def rnn(inputs, state, params):
    W_xh, H_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, H_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs , (H, )

#6 定义预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (Y, state) = rnn(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
        
    return ''.join([idx_to_char[i] for i in output])
# params = get_params()
# string = predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx)

#7.裁剪梯度
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

#8.困惑度

#9.定义模型训练函数
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, 
                            device, corpus_indices, idx_to_char, char_to_idx, is_random_iter, 
                            num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period,
                            pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
                else:
                    for s in state:
                        s.detach_()
                
            inputs = to_onehot(X, vocab_size)
            (outputs, state) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim=0)
