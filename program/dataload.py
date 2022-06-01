from tkinter.tix import Tree
from PIL import Image
import json
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import numpy as np
import matplotlib.pyplot as plt


json_files = {
    'train': '../FS2K/anno_train.json',
    'test': '../FS2K/anno_test.json'
}

# 数据预处理
train_transform = transforms.Compose([
    # transforms.ToPILImage(), # 转换数据格式为tensforms格式，才能进行后续处理
    transforms.Resize(128),# 按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
    # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
    # transforms.RandomHorizontalFlip(),# 把图像按照中心随机切割成224正方形大小的图片
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),# 转换为tensor格式，这个格式可以直接输入进神经网络
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 对像素值进行归一化处理
])

def add_suffix(file_path):
    if os.path.exists(file_path+'.jpg'):
        file_path += '.jpg'
    else:
        file_path += '.png'
    return file_path

class MyDataSet(torch.utils.data.Dataset):

    root_dir = '../FS2K'


    def image_load(self, photo_path):
        img = Image.open(photo_path)
        return self.transform(img)

    def __init__(self, json_file, train, transform, sketch):
        self.json_file = json_file
        self.train = train
        self.transform = transform
        self.sketch = sketch
        self.dataSet, self.labelSet = self.read()

    def read(self):

        if self.train == True:
            dir = "train"
        else:
            dir = "test"
        root_dir_spliltType = os.path.join(self.root_dir, dir)

        with open(self.json_file, 'r') as f:
            json_data = json.loads(f.read())
        
        dataSet = []
        labelSet= []
        for idx, item in enumerate(json_data):
            # image_name = item['image_name']
            # label = item[1:]

            # 属性
            image_name = item['image_name']
            image_name = image_name.replace('/image', '_image')
            skin_color = torch.tensor(item['skin_color'])
            lip_color = torch.tensor(item['lip_color'])
            eye_color = torch.tensor(item['eye_color'])
            hair = torch.tensor(item['hair'])
            hair_color = torch.tensor(item['hair_color'])
            gender = torch.tensor(item['gender'])
            earring = torch.tensor(item['earring'])
            smile = torch.tensor(item['smile'])
            frontal_face = torch.tensor(item['frontal_face'])
            style = torch.tensor(item['style'])

            # 照片和素描
            if self.sketch:
                sketch_path = add_suffix(os.path.join(root_dir_spliltType, 'sketch',\
                                    image_name.replace('photo', 'sketch').replace('image', 'sketch')))
                sketch = self.image_load(sketch_path)
                data = sketch
            else:
                photo_path = add_suffix(os.path.join(root_dir_spliltType, 'photo', image_name))
                photo = self.image_load(photo_path)
                data = photo

            # 输出
            labels = [skin_color, lip_color, eye_color, hair, hair_color, \
                      gender, earring, smile, frontal_face, style]
            dataSet.append(data)
            labelSet.append(labels)
        return dataSet, labelSet
    def __getitem__(self, index):
        data = self.dataSet[index]
        labels = self.labelSet[index]
        return data, labels
    
    def __len__(self):
        return len(self.dataSet)

batch_size = 64


trainset = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=False)
trainloader = dataloader.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=False)
testloader = dataloader.DataLoader(testset, batch_size=batch_size, shuffle=True)
