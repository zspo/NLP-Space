import random

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

# 定义字典
words_x = '<PAD>,1,2,3,4,5,6,7,8,9,0,<SOS>,<EOS>,+'
vocab_x = {word: i for i, word in enumerate(words_x.split(','))}
vocab_xr = [k for k, v in vocab_x.items()] #反查词典

words_y = '<PAD>,1,2,3,4,5,6,7,8,9,0,<SOS>,<EOS>'
vocab_y = {word: i for i, word in enumerate(words_y.split(','))}
vocab_yr = [k for k, v in vocab_y.items()] #反查词典

#两数相加数据集
def get_data():
    # 定义词集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 每个词被选中的概率
    p = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
    p = p / p.sum()

    # 随机采样n1个词作为s1
    n1 = random.randint(10, 20)
    s1 = np.random.choice(words, size=n1, replace=True, p=p)
    s1 = s1.tolist()

    # 随机采样n2个词作为s2
    n2 = random.randint(10, 20)
    s2 = np.random.choice(words, size=n2, replace=True, p=p)
    s2 = s2.tolist()

    # x等于s1和s2字符上的相加
    x = s1 + ['+'] + s2
    
    # y等于s1和s2数值上的相加
    y = int(''.join(s1)) + int(''.join(s2))
    y = list(str(y))
    
    # 加上首尾符号
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # 补pad到固定长度
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    # 编码成token
    token_x = [vocab_x[i] for i in x]
    token_y = [vocab_y[i] for i in y]

    # 转tensor
    tensor_x = torch.LongTensor(token_x)
    tensor_y = torch.LongTensor(token_y)
    return tensor_x, tensor_y


def show_data(tensor_x,tensor_y) ->"str":
    words_x = "".join([vocab_xr[i] for i in tensor_x.tolist()])
    words_y = "".join([vocab_yr[i] for i in tensor_y.tolist()])
    return words_x,words_y


# 定义数据集
class TwoSumDataset(torch.utils.data.Dataset):
    def __init__(self,size = 100000):
        super(Dataset, self).__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return get_data()
    
ds_train = TwoSumDataset(size = 100000)
ds_val = TwoSumDataset(size = 10000)


if __name__ == "__main__":

    x,y = get_data() 
    print(x,y,"\n")
    print(show_data(x,y)) 

    # 数据加载器
    dl_train = DataLoader(dataset=ds_train,
            batch_size=200,
            drop_last=True,
            shuffle=True)

    dl_val = DataLoader(dataset=ds_val,
            batch_size=200,
            drop_last=True,
            shuffle=False)
    
    for src,tgt in dl_train:
        print(src.shape)
        print(tgt.shape)
        break 
