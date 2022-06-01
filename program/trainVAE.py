from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from dataload import trainloader, testloader
import torch.utils.data.dataloader as dataloader

from AlexNet import AlexNet
from vae import VanillaVAE
from vae_copy import MyVAE

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Param:
    dropout_rate = 0.2
 
    predict_num_classes = 8
    classifier_num_classes = 44

    skin_patch_classes = 2
    lip_color_classes = 3
    eye_color_classes = 3
    hair_classes = 2
    hair_color_classes = 5
    gender_classes = 2
    earring_classes = 2
    smile_classes = 2
    frontal_face_classes = 31
    

Path = "../FS2K/model/vae_2048_sketch.pt"
# if __name__ == '__main__':




labelNO = 5
param = Param()
net = VanillaVAE(in_channels=3, latent_dim=2048)
# state_dict = torch.load(Path)
# net.load_state_dict(state_dict)
# print("模型参数加载完毕")
net.to(device)

classes = [
    'hair', 'hair_color', 'gender', 'earring', 'smile', 'frontal_face'
]


trainEpoch = 80
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

def imgShow(img):
    img = img.cpu()
    npimg = img.detach().numpy()
    for i in range(2):
        plt.subplot(1, 1, i+1)
        plt.imshow(np.transpose(npimg[i], (1, 2, 0)))
    plt.show()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))

choice = 'y'

totalEpoch = 0
while(choice == 'y'):
    for epoch in range(trainEpoch):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labelsSet = data
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            
            loss = net.loss_function(*outputs, M_N=0.005)

            # imgShow(outputs[0])
            # imgShow(inputs)
            running_loss += loss.item()
            
            loss.backward()    
            optimizer.step()

            print("\r", end="")
            print("[epoch %3d] Training progress: " % (totalEpoch + epoch + 1), "▋" * int(i * 1.0 / len(trainloader) * 100), end="")
        print("\r", ' '* 200, end="")     
        print('\r[epoch %3d] loss: %3f' % (totalEpoch + epoch + 1, running_loss / len(trainloader)))
        
        # correctSet = [0] * 6
        # totalSet = [0] * 6
        # # 测试损失

        # with torch.no_grad():
        #     dataiter = iter(testloader)
        #     inputs, labelsSet = data
        #     inputs = inputs.to(device)

        #     outputs = net(inputs)
        #     imgShow(outputs[0])
        #     imgShow(inputs)
        # # 进行测试
        # with torch.no_grad():
        #     for i, data in enumerate(testloader, 0):
        #         inputs, labelsSet = data
        #         inputs = inputs.to(device)
        #         for labels in labelsSet:
        #             labels.to(device) 

        #         # 测试数据
        #         outputsSet = net(inputs)
                
        #         for i in range(len(outputsSet)):
        #             outputs = outputsSet[i]
        #             correct = correctSet[i]
        #             total = totalSet[i]
        #             # 找到概率最大的下标
        #             _, predicted = torch.max(outputs.data, 1)
        #             # 统计正确率
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()
            

        #     print("Test epoch : [ %4d/%4d ]" % \
        #          (epoch, totalEpoch))
            
        #     for i in range(len(running_losses)):
        #         name = classes[i]
        #         correct = correctSet[i]
        #         total = totalSet[i]
        #         running_loss = running_losses[i]
        #         print("\t%s 测试结果: \
        #                \t\t Running_loss: %.4f\
        #                \t\t Accuracy: %.4f %%"
        #                %(running_loss / len(trainloader, 100 * correct / total)))
    totalEpoch += trainEpoch
    print("Finished Training")

    # with torch.no_grad():
    #     dataiter = iter(testloader)
    #     inputs, labelsSet = data
    #     inputs = inputs.to(device)

    #     outputs = net(inputs)
    #     imgShow(outputs[0])
    #     imgShow(inputs)

    choice = input("继续？(y/n):")     
torch.save(net.state_dict(), Path)
print("模型保存完毕")
