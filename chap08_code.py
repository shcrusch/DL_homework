import numpy as np
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import string
import re
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

rng = np.random.RandomState(1234)
random_state = 42


def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text

# 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）


def tokenizer_punctuation(text):
    return text.strip().split()


# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


# 文章とラベルの両方に用意します
max_length = 256
TEXT = data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                  use_vocab=True, lower=True, include_lengths=True,
                  batch_first=True, fix_length=max_length,
                  init_token="<cls>", eos_token="<eos>")
LABEL = data.Field(sequential=False, use_vocab=True)

# 引数の意味は次の通り
# init_token：全部の文章で、文頭に入れておく単語
# eos_token：全部の文章で、文末に入れておく単語


# データセットの作成

train_data, test_data = datasets.IMDB.splits(text_field=TEXT,
                                             label_field=LABEL)

for i in range(len(test_data)):
    if i % 2 == 0:
        test_data[i].label = "pos"
    else:
        test_data[i].label = "neg"


word_num = 5000
TEXT.build_vocab(train_data, max_size=word_num)
LABEL.build_vocab(train_data)

batch_size = 100

train_dl = torchtext.data.Iterator(train_data, batch_size=batch_size,
                                   train=True, sort=True)

valid_dl = torchtext.data.Iterator(valid_data, batch_size=batch_size,
                                   train=False, sort=False)

test_dl = torchtext.data.Iterator(test_data, batch_size=batch_size,
                                  train=False, sort=False)


def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

class SequenceTaggingNet4(nn.Module):
    def __init__(self, word_num, emb_dim, hid_dim):
        super().__init__()
        self.Emb = nn.Embedding(word_num, emb_dim)
        self.RNN = nn.LSTM(emb_dim, hid_dim, 6, batch_first=True)  # nn.LSTMの使用
        self.Linear = nn.Linear(hid_dim, 1)
    
    def forward(self, x, len_seq_max=0, len_seq=None, init_state=None):
        h = self.Emb(x)
        if len_seq_max > 0:
            h, _ = self.RNN(h[:, 0:len_seq_max, :], init_state)
        else:
            h, _ = self.RNN(h, init_state)
        h = h.transpose(0, 1)
        if len_seq is not None:
            h = h[len_seq - 1, list(range(len(x))), :]
        else:
            h = h[-1]
        y = self.Linear(h)
        
        return y

emb_dim = 400
hid_dim = 200
n_epochs = 15
device = 'cuda'

net = SequenceTaggingNet4(word_num + 4, emb_dim, hid_dim)
net.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.002)

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []

    net.train()
    n_train = 0
    acc_train = 0
    for mini_batch in train_dl:
        net.zero_grad()  # 勾配の初期化

        t = mini_batch.label.to(device)-1  # テンソルをGPUに移動
        x = mini_batch.text[0].to(device)
        len_seq = mini_batch.text[1].to(device)
        h = net(x, torch.max(len_seq), len_seq)
        y = torch.sigmoid(h).squeeze()
      
        loss = -torch.mean(t*torch_log(y) + (1 - t)*torch_log(1 - y))

        loss.backward()  # 誤差の逆伝播

        # 勾配を絶対値1.0でクリッピングする
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        
        optimizer.step()  # パラメータの更新

        losses_train.append(loss.tolist())

        n_train += t.size()[0]

    # Valid
    t_valid = []
    y_pred = []
    net.eval()
    for mini_batch in valid_dl:
        t = mini_batch.label.to(device)-1  # テンソルをGPUに移動
        x = mini_batch.text[0].to(device)
        len_seq = mini_batch.text[1].to(device)
        h = net(x, torch.max(len_seq), len_seq)
        y = torch.sigmoid(h).squeeze()
        
        loss = -torch.mean(t*torch_log(y) + (1 - t)*torch_log(1 - y))

        pred = y.round().squeeze()  # 0.5以上の値を持つ要素を正ラベルと予測する

        t_valid.extend(t.tolist())
        y_pred.extend(pred.tolist())

        losses_valid.append(loss.tolist())

    print('EPOCH: {}, Train Loss: {:.3f}, Valid Loss: {:.3f}, Validation F1: {:.3f}'.format(
        epoch,
        np.mean(losses_train),
        np.mean(losses_valid),
        f1_score(t_valid, y_pred, average='macro')
    ))