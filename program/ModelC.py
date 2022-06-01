import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
import numpy as np
import cv2

from dataLoadForMyModel import trainloaderForPhotoC, trainloaderForSketchC, testloaderForPhotoC, testloaderForSketchC

# 判断使用CPU训练还是GPU训练
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置训练数据集的轮次
EPOCHS = 10
# 设置学习率
learning_rate = 1e-4

# 设置每个分类任务的名称
labels = ['skin color', 'lip color', 'eye color', 'hair', 'hair color', 'gender', 'earring', 'smile', 'frontal face',
          'style']
# 设置每个分类任务的类别总数
labelNum = [2, 3, 3, 2, 5, 2, 2, 2, 2, 3]


# 网络模型
class Digit(nn.Module):
    def __init__(self, labelNum):
        super().__init__()
        # 二维卷积层1
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),  # 批归一化
            nn.Conv2d(3, 10, 5, padding=2),  # 3：图片输入通道为3，10：输出通道，5：5*5卷积核
            nn.BatchNorm2d(10),  # 批归一化
            nn.LeakyReLU()  # 激活函数
        )
        # 二维卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 3, padding=1),  # 10：输入通道，20：输出通道，3：3*3卷积核
            nn.BatchNorm2d(20),  # 批归一化
            nn.Dropout(p=0.2),  # 使神经元随机失效
            nn.LeakyReLU()
        )
        # 二维卷积层3
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),  # 20：输入通道，20：输出通道，3：3*3卷积核
            nn.BatchNorm2d(20),  # 批归一化
            nn.Dropout(p=0.2),  # 使神经元随机失效
            nn.LeakyReLU()
        )
        # 二维卷积层4
        self.conv4 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),  # 20：输入通道，20：输出通道，3：3*3卷积核
            nn.BatchNorm2d(20),  # 批归一化
            nn.Dropout(p=0.2),  # 使神经元随机失效
            nn.LeakyReLU()
        )
        # 二维转置卷积1
        self.convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(20),
            nn.Dropout(p=0.2),
            nn.LeakyReLU()
        )
        # 二维转置卷积2
        self.convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(10),
            nn.Dropout(p=0.2),
            nn.LeakyReLU()
        )
        # 全连接层1
        self.fc1 = nn.Linear(10 * 64 * 64, 5000)  # 10*64*64：输入通道，5000：输出通道
        # 全连接层2
        self.fc2 = nn.Linear(5000, 200)  # 5000：输入通道，200：输出通道
        # 全连接层3
        self.fc3 = nn.Linear(200, labelNum)  # 200：输入通道，labelNum：输出通道

    def forward(self, x, labelNum):
        input_size = x.size(0)  # 其实就是batch_size

        x = self.conv1(x)  # 输入：batch_size*3*256*256，输出：batch_size*10*256*256（其中256=256-5+1+4）
        x = F.max_pool2d(x, 2, 2)  # 池化层（数据压缩，降采样），输入：batch_size*10*256*256，输出：batch_size*10*128*128

        x = self.conv2(x)  # 输入：batch_size*10*128*128，输出：batch_size*20*128*128（其中128=128-3+1+2）
        x = F.max_pool2d(x, 2, 2)  # 池化层（数据压缩，降采样），输入：batch_size*20*128*128，输出：batch_size*20*64*64

        x = self.conv3(x)  # 输入：batch_size*20*64*64，输出：batch_size*20*64*64（其中64=64-3+1+2）
        x = F.max_pool2d(x, 2, 2)  # 池化层（数据压缩，降采样），输入：batch_size*20*64*64，输出：batch_size*20*32*32

        x = self.conv4(x)  # 输入：batch_size*20*32*32，输出：batch_size*20*32*32（其中32=32-3+1+2）
        x = F.max_pool2d(x, 2, 2)  # 池化层（数据压缩，降采样），输入：batch_size*20*32*32，输出：batch_size*20*16*16

        x = self.convtrans1(x)  # 输入：batch_size*20*16*16，输出：batch_size*20*32*32
        x = self.convtrans2(x)  # 输入：batch_size*20*32*32，输出：batch_size*10*64*64

        x = x.view(input_size, -1)  # 降维到1维，-1：自动计算原始维度

        x = self.fc1(x)  # 输入：batch_size*40*64*64，输出：batch_size*5000
        x = F.relu(x)  # 激活函数,保持形状不变，输出：batch_size*5000

        x = self.fc2(x)  # 输入：batch_size*5000，输出：batch_size*200
        x = F.relu(x)  # 激活函数,保持形状不变，输出：batch_size*200

        x = self.fc3(x)  # 输入：batch_size*200，输出：batch_size*labelNum
        output = F.log_softmax(x, dim=1)  # 计算分类后每个数字的概率值，dim=1：按行
        return output


def train_model(model, device, train_loader, optimizer, epoch, labelNO):
    # 模型训练
    model.train()
    for batch_index, (data, label) in enumerate(train_loader):
        label = label[labelNO]
        # 部署到DEVICE上
        data, label = data.to(device), label.to(device)
        # 初始化梯度为0
        optimizer.zero_grad()
        # 训练结果
        output = model(data, labelNum[labelNO])
        # 计算损失
        loss = F.cross_entropy(output, label)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 查看损失
        if batch_index % 3000 == 0:
            print("Train Epoch:{}\t Loss:{:.6f}".format(epoch, loss.item()))


def test_model(model, device, test_loader, labelNO):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    # 进行测试
    with torch.no_grad():
        labelName = labels[labelNO]
        for data, label in test_loader:
            label = label[labelNO]
            # 部署到DEVICE上
            data, label = data.to(device), label.to(device)
            # 测试数据
            output = model(data, labelNum[labelNO])
            # 计算测试损失
            test_loss += F.cross_entropy(output, label)
            # 找到概率最大的下标
            pred = output.max(1, keepdim=True)[1]  # 返回值格式为（值，索引）
            # 统计正确率
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        correct /= len(test_loader.dataset)
        print("Test —— Average Loss:{:.4f}, 标签{}的准确率为{:.3f}%".format(
            test_loss, labelName, correct * 100))
        return correct * 100


# 调用训练方法
def trainAndTest(labelNo, type, result):
    for epoch in range(1, EPOCHS + 1):
        model = Digit(labelNum[labelNo]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if type == 'photo':
            train_model(model, DEVICE, trainloaderForPhotoC, optimizer, epoch, labelNo)
            correct = test_model(model, DEVICE, testloaderForPhotoC, labelNo)
            if epoch == EPOCHS:
                result.append({'数据类型': 'photo', '分类用的标签': labels[labelNo], '分类正确率': correct})
        elif type == 'sketch':
            train_model(model, DEVICE, trainloaderForSketchC, optimizer, epoch, labelNo)
            correct = test_model(model, DEVICE, testloaderForSketchC, labelNo)
            if epoch == EPOCHS:
                result.append({'数据类型': 'sketch', '分类用的标签': labels[labelNo], '分类正确率': correct})
        else:
            print('this type of dataset is not exist!')


if __name__ == '__main__':
    labelsToUse = [3, 4, 5, 6, 7, 8, 9]
    types = ['photo', 'sketch']
    result = []
    for type in types:
        for label in labelsToUse:
            trainAndTest(label, type, result)
    s = ''
    for item in result:
        for (key, value) in item.items():
            s = s + key + ':' + str(value) + ' '
        s += '\n'
    print(s)
