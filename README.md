# -介绍

本项目使用的数据集是[F2Sk](https://github.com/DengPingFan/FS2K)，数据集的有关介绍请参阅F2Sk的github项目介绍。本项目使用了5种模型对F2Sk数据集中的hair、hair_color、gender、earring、smile、frontal_face和style这7个任务进行训练和测试。

# -安装依赖

克隆本项目后，可以使用以下命令安装依赖：

1.如果安装了python3，则跳过此步骤，否则：

```shell
sudo apt update
sudo apt install python3
```

2.使用以下命令安装项目所需的依赖包：

```shell
cd DLAssignment-main
pip3 install -r requirements.txt
```

# -使用

1. 使用以下命令可以调用模型A来完成分类任务：

```shell
python3 ./program/ModelA.py
```


2.使用以下命令可以调用模型B来完成分类任务：

```shell
python3 ./program/ModelB.py
```

3.使用以下命令可以调用模型C来完成分类任务：

```shell
python3 ./program/ModelC.py
```

4.使用以下命令可以调用VAE+Classifier来完成分类任务：

```shell
# 训练vae模型
python3 ./program/trainVAE.py
# 训练分类器
python3 ./program/trainVaeClassifier.py
```

5.使用以下命令可以调用AlexNet来完成分类任务：

```shell
python3 ./program/trainAlexNet.py
```



