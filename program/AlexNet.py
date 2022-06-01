import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import init



class AlexNet(nn.Module):
    def __init__(self, global_params=None):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        # self.predict = nn.Sequential(
        #     nn.Dropout(global_params.dropout_rate),
        #     nn.Linear(256 * 7 * 7, 4096),
        #     nn.Dropout(global_params.dropout_rate),
        #     nn.Linear(4096, 4096),
        #     nn.Linear(4096, global_params.predict_num_classes),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(global_params.dropout_rate),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(global_params.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, global_params.classifier_num_classes),
        )

        # for name, param in self.named_parameters():
        #     init.normal_(param, mean= 0., std=0.1)


    def forward(self, inputs):
        # See note [TorchScript super()]
        x = self.features(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        # 256*49
        # predict = self.predict(x)
        classifier = self.classifier(x)
        return classifier

