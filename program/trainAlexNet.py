from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np

from dataload import trainloader, testloader
import torch.utils.data.dataloader as dataloader

from AlexNet import AlexNet

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Param:
    dropout_rate = 0.2

    predict_num_classes = 8
    classifier_num_classes = 3

    skin_patch_classes = 2
    lip_color_classes = 3
    eye_color_classes = 3
    hair_classes = 2
    hair_color_classes = 5
    gender_classes = 2
    earring_classes = 2
    smile_classes = 2
    frontal_face_classes = 2
    
labelNO = 9
classes = [
    'hair', 'hair_color', 'gender', 'earring', 'smile', 'frontal_face', "style"
]
# if __name__ == '__main__':
Path = "../FS2K/model/"
task = "alex_" + classes[labelNO -3] + ".pt"

totalEpoch = 30

param = Param()
net = AlexNet(param)
net.to(device)



criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.Adam(net.parameters())


for epoch in range(totalEpoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labelsSet = data
        inputs = inputs.to(device)
        labels = labelsSet[labelNO].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # print(outputs.data)
        # print(labels.data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 200 == 199:
        #     print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))

    
    correct = 0
    total = 0
    # 测试损失

    # 进行测试
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labelsSet = data
            inputs = inputs.to(device)
            labels = labelsSet[labelNO].to(device)

            # 测试数据
            outputs = net(inputs)
            
            # 找到概率最大的下标
            _, predicted = torch.max(outputs.data, 1)

            # print(predicted)
            # 统计正确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("\r", end="")
            print("[epoch %3d] Training progress: " % (epoch + 1), "▋" * int(i * 1.0 / len(trainloader) * 100), end="")
        print("\r", ' '* 200, end="")     
        print("\rTest epoch : [ %4d/%4d ]" % \
                (epoch + 1, totalEpoch))

        print("\t测试结果:\n\t\t Running_loss: %.4f\n\t\t Accuracy: %.4f"%(running_loss / len(trainloader), 100. * correct / total))

print("Finished Training")
if input("保存？(y/n):") == 'y':     
    torch.save(net.state_dict(), os.path.join(Path, task))