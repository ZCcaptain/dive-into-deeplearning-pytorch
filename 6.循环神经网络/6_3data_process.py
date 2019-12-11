import torch
import random
import zipfile

#1.读取数据集
with zipfile.ZipFile('./jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[:10000]

#2.建立字符索引
idx_to_char = list(set(corpus_chars))
char_to_idx = dict((char, i) for i, char in enumerate(idx_to_char))
vocab_size = len(char_to_idx)

corpus_indices = [char_to_idx[ch] for ch in corpus_chars]

#3.时序数据的采样
#3.1.随机采样
#在随机采样中，每个样本是原始序列上任意截取的⼀段序列。相邻的两个随机⼩批量在原始序列上的位置不⼀定相毗邻。因此，我们⽆法⽤⼀个⼩批量最终
#时间步的隐藏状态来初始化下⼀个⼩批量的隐藏状态。
def data_iter_random(corpus_indices, batch_size, num_steps, device = None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos:pos+num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i+batch_size]
        x = [_data(j * num_steps) for j in batch_indices]
        y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(x, dtype=torch.float32, device=device),torch.tensor(y, dtype=torch.float32, device=device)
    
my_seq = list(range(30))

# for X, Y in data_iter_random(my_seq, 2, 6):
#     print(X, Y)
    
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device = device)
        data_len = len(corpus_indices)
        batch_len = data_len // batch_size
        indices = corpus_indices[0:batch_size*batch_len].view(batch_size, batch_len)
        epoch_size = (batch_len - 1) // num_steps
        for i in range(epoch_size):
            i = i * num_steps
            x = indices[:, i:i+num_steps]
            y = indices[:, i + 1:i + num_steps + 1]
            yield x, y

for X, Y in data_iter_consecutive(my_seq, 2, 6):
  print('X: ', X, '\nY:', Y, '\n')