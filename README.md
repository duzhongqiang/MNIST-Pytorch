# pytorch手写数字识别

## 项目介绍
本项目使用pytorch实现了简单的手写数字识别，运用了三种方法
### 1.两层全连接网络
### 2.四层全连接网络
### 3.四层卷积神经网络
通过在代码中修改相应的参数，即可选择对应的方法，参考注释

## 文件介绍
### displayMNIST.py是显示手写数字识别（NMIST）数据集
### mnistCPU.py是通过CPU进行训练
### mnistGPU.py是通过GPU进行训练

## 运行方式
 先直接运行mnistCPU.py或者mnistGPU.py就可以下载数据集，数据集下载成功后，就开始了训练，如果要查看数据集，对数据集进行显示，可以运行displayMNIST.py代码
## 环境要求
python3，pytorch
