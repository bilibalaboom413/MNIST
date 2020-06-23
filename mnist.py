import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

torch.manual_seed(1)  # 设置随机种子；可复现性
# torch.cuda.set_device(gpu_id)
# 超参数
EPOCH = 1
LR = 0.001
BATCH_SIZE = 50
DOWNLOAD_MNIST = False

# mnist手写数据集
# 训练集
train_data = torchvision.datasets.MNIST(  # torchvision中有这一数据集，可以直接下载
    root='./MNIST/',  # 下载后存放位置
    train=True,  # train如果设置为True，代表是训练集
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 [0.0,255.0] normalize 成 [0.0, 1.0]区间
    download=DOWNLOAD_MNIST  # 是否下载；如果已经下载好，之后就不必下载
)

# 测试集
test_data = torchvision.datasets.MNIST(
    root='./MNIST/', train=False)  # train设置为False表示获取测试集

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

#  测试数据预处理；只测试前2000个
test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000].cuda() / 255.0  # 一维转为二维
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # b, 16, 26, 26  # 如果想要 con2d 出来的图片长宽没有变化,当 stride=1, padding=(kernel_size-1)/2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # b, 32, 12, 12  （24-2）/2+1 stride默认等于k_size

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 128, 4, 4
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # 展平，二维变成一维
        x = self.fc(x)
        return x

cnn = CNN()
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x.cuda())
        loss = loss_func(pred_y, batch_y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            # torch.max(test_out,1)返回的是test_out中每一行最大的数)
            pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().numpy()
            # 返回的形式为torch.return_types.max(
            #           values=tensor([0.7000, 0.9000]),
            #           indices=tensor([2, 2]))
            # 后面的[1]代表获取indices
            accuracy = float((pred_y == test_y.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))  # astype()数据类型转换；True 1; False 0
            print('Epoch: ', epoch, '| train loss: %.4f' %loss.data, '| test accuracy: %.2f' %accuracy)

# 打印前十个测试结果和真实结果进行对比
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].cpu().numpy(), 'real number')

