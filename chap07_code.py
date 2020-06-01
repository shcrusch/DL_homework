from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

x_train, x_valid, t_train, t_valid = train_test_split(x_trainval, t_trainval, test_size=0.1)
train_data = train_dataset(x_train, t_train)
valid_data = train_dataset(x_valid, t_valid)
test_data = test_dataset(x_test)

# .npy型のデータを読み込む関係で演習の時とは処理が多少変わっていますが、
# 本質的な計算処理は変わりません。

class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):  # 計算が重いのでGPUを用いる
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):  # 変換行列と平均をデータから計算
        x = torch.tensor(images[0].reshape(1, -1))
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # 各データについての平均を取る
            x = torch.tensor(images[i].reshape(1, -1)).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        E, V = torch.symeig(con_matrix, eigenvectors=True)  # 固有値分解
        self.ZCA_matrix = torch.mm(torch.mm(V, torch.diag((E.squeeze()+self.epsilon)**(-0.5))), V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x

zca = ZCAWhitening()
zca.fit(train_data.x_data)

# 前処理を定義
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=(4, 4, 4, 4), padding_mode='constant'),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      zca
                                      ])

transform = transforms.Compose([transforms.ToTensor(),
                                zca
                                ])

train_data.transform = transform_train
valid_data.transform = transform
test_data.transform = transform

batch_size = 32

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False
)

dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)



rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


conv_net = nn.Sequential(
    nn.Conv2d(3, 8, 3),              # 32->30
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.AvgPool2d(2),     #30->15
    nn.Conv2d(8, 16, 3),             # 15-13
    nn.BatchNorm2d(16),    #
    nn.ReLU(),
    nn.AvgPool2d(2),                  # 13-6
    nn.Conv2d(16, 32, 3),            # 6 -> 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.AvgPool2d(2),                  # 4 -> 2
    nn.Flatten(),
    nn.Linear(2*2*32, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)


def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)


conv_net.apply(init_weights)


n_epochs = 5
lr = 0.01
device = 'cuda'

conv_net.to(device)
optimizer = optim.Adam(conv_net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []

    conv_net.train()
    n_train = 0
    acc_train = 0
    for x, t in dataloader_train:
        n_train += t.size()[0]

        conv_net.zero_grad()  # 勾配の初期化

        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)

        y = conv_net.forward(x)  # 順伝播

        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算

        loss.backward()  # 誤差の逆伝播

        optimizer.step()  # パラメータの更新

        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする


        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())

    conv_net.eval()
    n_val = 0
    acc_val = 0
    for x, t in dataloader_valid:
        n_val += t.size()[0]

        x = x.to(device)  # テンソルをGPUに移動
        t = t.to(device)

        y = conv_net.forward(x)  # 順伝播

        loss = loss_function(y, t)  # 誤差(クロスエントロピー誤差関数)の計算

        pred = y.argmax(1)  # 最大値を取るラベルを予測ラベルとする


        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())

    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
        epoch,
        np.mean(losses_train),
        acc_train/n_train,
        np.mean(losses_valid),
        acc_val/n_val
    ))