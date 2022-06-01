import cv2
import json
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import numpy as np
import matplotlib.pyplot as plt

json_files = {
    'train': '../anno_train.json',
    'test': '../anno_test.json'
}

# 数据预处理
train_transform = transforms.Compose([
    transforms.ToPILImage(),  # 转换数据格式为tensforms格式，才能进行后续处理
    transforms.Resize(256),  # 按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
    # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
    # transforms.RandomHorizontalFlip(),# 把图像按照中心随机切割成224正方形大小的图片
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),  # 转换为tensor格式，这个格式可以直接输入进神经网络
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 对像素值进行归一化处理
])


def add_suffix(file_path):
    if os.path.exists(file_path + '.jpg'):
        file_path += '.jpg'
    else:
        file_path += '.png'
    return file_path


class MyDataSet(torch.utils.data.Dataset):
    root_dir = '..'

    def image_load(self, photo_path):
        img = cv2.imread(photo_path)
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
        labelSet = []
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
                sketch_path = add_suffix(os.path.join(root_dir_spliltType, 'sketch', \
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


trainsetForPhotoA = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=False)
trainloaderForPhotoA = dataloader.DataLoader(trainsetForPhotoA, batch_size=64, shuffle=True)
trainsetForSketchA = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=True)
trainloaderForSketchA = dataloader.DataLoader(trainsetForSketchA, batch_size=64, shuffle=True)

testsetForPhotoA = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=False)
testloaderForPhotoA = dataloader.DataLoader(testsetForPhotoA, batch_size=64, shuffle=True)
testsetForSketchA = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=True)
testloaderForSketchA = dataloader.DataLoader(testsetForSketchA, batch_size=64, shuffle=True)

trainsetForPhotoB = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=False)
trainloaderForPhotoB = dataloader.DataLoader(trainsetForPhotoB, batch_size=16, shuffle=True)
trainsetForSketchB = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=True)
trainloaderForSketchB = dataloader.DataLoader(trainsetForSketchB, batch_size=16, shuffle=True)

testsetForPhotoB = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=False)
testloaderForPhotoB = dataloader.DataLoader(testsetForPhotoB, batch_size=16, shuffle=True)
testsetForSketchB = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=True)
testloaderForSketchB = dataloader.DataLoader(testsetForSketchB, batch_size=16, shuffle=True)

trainsetForPhotoC = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=False)
trainloaderForPhotoC = dataloader.DataLoader(trainsetForPhotoC, batch_size=4, shuffle=True)
trainsetForSketchC = MyDataSet(json_files['train'], train=True, transform=train_transform, sketch=True)
trainloaderForSketchC = dataloader.DataLoader(trainsetForSketchC, batch_size=4, shuffle=True)

testsetForPhotoC = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=False)
testloaderForPhotoC = dataloader.DataLoader(testsetForPhotoC, batch_size=4, shuffle=True)
testsetForSketchC = MyDataSet(json_files['test'], train=False, transform=train_transform, sketch=True)
testloaderForSketchC = dataloader.DataLoader(testsetForSketchC, batch_size=4, shuffle=True)

