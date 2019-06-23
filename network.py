# 网络模型搭建
# 基于mxnet框架
import pickle
import time
import os
import shutil
import numpy as np
import pandas as pd

import mxnet as mx
from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import autograd, gluon, init, nd

# AlexNet搭建
def Alex_build(label_type=5, ratio=0.5):
    net = nn.Sequential()
    net.add(
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=(2, 2), strides=2),
        nn.Conv2D(channels=256, kernel_size=5, strides=1,padding=2,  activation='relu'),
        nn.MaxPool2D(pool_size=(3, 3), strides=2),
        nn.Conv2D(channels=384, kernel_size=3, strides=1, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, strides=1, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=(3, 3), strides=2),
        nn.Dense(units=4096, activation='relu'),
        nn.Dropout(ratio),
        nn.Dense(units=4096, activation='relu'),
        nn.Dropout(ratio),
        nn.Dense(units=label_type)
    )
    return net



# 读取标签, 返回总数据集大小，分割测试集大小，以及以字典形式表示的标签
def read_label(label_path='.\dataset\HAND POSE', test_ratio=0.2):
    with open(label_path + '\\' + 'labels.csv', 'r') as f:
        # 跳过栏名称行
        lines = f.readlines()[1:]
        # 按，分割并去除每行末尾的\n
        tokens = [l.rstrip().split(',') for l in lines]
        # 删除每个l里第一个元素(csv索引)
        for l in tokens:
            del l[0]
        
    # 转换为dict
    idx_label = dict(((int(idx), label) for idx, label in tokens))
    # 提取label种类
    labels = set(idx_label.values())
    # 确定数据集大小并分割出测试集长度
    n_train_test = len(tokens)
    n_train = int(n_train_test * (1 - test_ratio))
    return n_train_test, n_train // len(labels), idx_label   # 返回的idx_label为字典格式


# 当路径不存在时，创建路径
def mkdir_if_not_exist(path):
    if os.path.exists(path) == False:
        print(*path, " not exist, try to mkdir")
        os.makedirs(path)

# 分割原始手势图片集为训练集和测试集，并保存于.\dataset\HAND POSE\train和.\dataset\HAND POSE\test下
def reorg_train_test(idx_of_label, n_train_per_label, data_path='.\dataset\HAND POSE'):
    label_count = {}
    for data_file in os.listdir(data_path + '\\'):
        if data_file == 'labels.csv':
            continue
        else:
            idx = int(data_file.split('.')[0])
            label = idx_of_label[idx]
            # print(idx, label)
            mkdir_if_not_exist(('.\dataset\\train')) # 检查train文件夹是否存在
            mkdir_if_not_exist(('.\dataset\\train')) # 检查test文件夹是否存在
    
            if label not in label_count or label_count[label] < n_train_per_label:
                mkdir_if_not_exist(('.\dataset\\train' + '\\' + label))
                shutil.copy(data_path + '\\' + data_file, '.\dataset\\train' + '\\' + label)
                label_count[label] = label_count.get(label, 0) + 1
            else:
                mkdir_if_not_exist(('.\dataset\\test' + '\\' + 'unknown')) 
                shutil.copy(data_path + '\\' + data_file, '.\dataset\\test' + '\\' + 'unknown')
                
# 整理手势数据
def manage_pose(data_path='.\dataset\HAND POSE'):
    n_train_test, n_test, idx_label = read_label()
    reorg_train_test(idx_of_label=idx_label, n_train_per_label=n_test, data_path='.\dataset\HAND POSE')

# 读取手势数据
def read_pose(batch_size=10, data_path='.\dataset'):
    # 图像增广
    transform_train = gdata.vision.transforms.Compose([
        # 随机翻转
        gdata.vision.transforms.RandomFlipLeftRight(),
        # 随机变化亮度、对比度和饱和度
        gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # 随机加噪声
        gdata.vision.transforms.RandomLighting(0.1),
        gdata.vision.transforms.ToTensor(),
        # 通道标准化
        gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = gdata.vision.transforms.Compose([
        gdata.vision.transforms.ToTensor(),
        # 通道标准化
        gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # 创建ImageFolderDataset实例来读取整理后的含原始图像文件的数据集，每个数据样本包括图像与标签, flag=1说明输入图像有三个通道
    # 注意文件夹结构
    train_ds = gdata.vision.ImageFolderDataset(data_path + '\\' + 'train', flag=1)
    test_ds = gdata.vision.ImageFolderDataset(data_path + '\\' + 'test', flag=1)
    # 创建dataloader
    train_iter = gdata.DataLoader(train_ds.transform_first(transform_train), batch_size, shuffle=True, last_batch='keep')
    test_iter = gdata.DataLoader(train_ds.transform_first(transform_test), batch_size, shuffle=False, last_batch='keep')
    return train_iter, test_iter

# 是否适用GPU
def try_gpu():
    try:
        ctx = mx.gpu()
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

# 获得测试准确率 
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum = (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

# 训练网络参数
def train(net_used=Alex_build(), batch_size=10, lr=0.01, epoch=5, ctx=mx.cpu()):    
    print("training on", ctx)
    print()
    train_iter, test_iter = read_pose(batch_size=batch_size)
    time.sleep(0.5)
    
    # 初始化训练参数，选择优化方法和损失函数
    train_acc, test_acc = 0, 0  
    net = net_used
    net.initialize(ctx=ctx, init=init.Xavier())
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(params=net.collect_params(), optimizer='adam', optimizer_params={'learning_rate': lr})
    
    # 开始训练
    print("start training")
    for i in range(epoch):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        
        test_acc = evaluate_accuracy(data_iter=test_iter, net=net, ctx=ctx)
        print('epoch: %d, loss: %.4f, train_acc: %.3f, test_acc: %.3f' % (i + 1, train_l_sum/n, train_acc_sum / n, test_acc))

    return net

# 显示网络
def show_network(net):
    print(net)

# 保存训练得到的网络参数
def save_params(net):
    net.save_parameters('.\\net_parameter' + '\\' + 'pose_params')

# 导入训练得到的网络参数
def read_params(net=Alex_build()):
    net_read = net
    net_read.load_parameters('.\\net_parameter' + '\\' + 'pose_params')
    return net_read

# network构建主程序
def network_main(net_used=Alex_build(), batch_size=10, lr=0.01, epoch=5, ctx=mx.cpu()):
    # 数据集导入
    print("start reading data")
    print()
    manage_pose()
    time.sleep(0.5)
    # 开始训练
    print("start training")
    print()
    time.sleep(0.5)
    return train(net_used=net_used, batch_size=batch_size, lr=lr, epoch=epoch, ctx=mx.cpu())


if __name__ == '__main__':
    # 设定超参数
    label_type = 5
    batch_size, lr, epoch, ratio = 1, 0.01, 10, 0.5
    # print(network_main(net_used=Alex_build(label_type=label_type, ratio=ratio), batch_size=batch_size, lr=lr, epoch=epoch))
    # 实际训练并保存网络数据
    # save_params(network_main(net_used=Alex_build(label_type=label_type, ratio=ratio), batch_size=batch_size, lr=lr, epoch=epoch))
    train_iter, test_iter = read_pose(batch_size=1, data_path='.\dataset')
    for X, y in test_iter:
        # print("X: ", X)
        # print("y: ", y)