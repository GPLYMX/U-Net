import random
import math
import yaml
from pathlib import Path
import sys
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.network import AttU_Net, U_Net
from utiles.utiles import CustomSegmentationDataset
from utiles.dice_score import dice_coeff, multiclass_dice_coeff, dice_loss



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('use_GPU:', True)
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(3407)
else:
    print('use_GPU:', False)
    device = torch.device('cpu')
    torch.manual_seed(3407)
random.seed(3407)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


configs = load_config('configs.yaml')
num_classes = configs['num_classes']
in_channels = configs['in_channels']
batch_size = configs['batch_size']
num_epochs = configs['num_epochs']


def clear_catch(n=10):
    for i in range(n):
        torch.cuda.empty_cache()


best_dice = 0.5

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1., 3.]))
model = AttU_Net(img_ch=len(in_channels), output_ch=num_classes)
# model.load_state_dict(torch.load('temp.pt'))

optimizer = optim.Adam(model.parameters(), lr=0.0001)
model = model.to(device)


def l1_regularization(model, l1_alpha=0.00001):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha=0.000003):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


# 在训练循环中进行训练
for epoch in range(num_epochs):
    model.train()
    model = model.to(device)
    total_loss = 0
    dice_score = 0
    n = 0

    train_dataset = CustomSegmentationDataset(data_dir=r'../data_20230921/train',
                                              channels=in_channels, label_format='png', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CustomSegmentationDataset(data_dir=r'../data_20230921/test',
                                             channels=in_channels, label_format='png', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 训练
    for batch_sample, batch_label in train_loader:
        model.train()
        n = n + 1
        optimizer.zero_grad()  # 梯度清零

        base_h, base_w = configs['base_size']
        height, width = batch_sample.shape[2], batch_sample.shape[3]
        iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
        boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
        padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
                base_w - boundary_pixel_num_w) % base_w

        batch_sample = torch.nn.functional.pad(batch_sample, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                               mode='replicate').to('cpu')
        batch_label = torch.nn.functional.pad(batch_label, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                              mode='replicate').to('cpu')

        loss = 0
        for i in range(iter_num_h):
            for j in range(iter_num_w):
                model.to(device)
                offset_h, offset_w = i * base_h, j * base_w
                temp_sample = batch_sample[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                temp_label = batch_label[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]

                outputs = model(temp_sample.to(device))
                temp_sample.to('cpu')
                # 计算损失
                loss = criterion(outputs.to('cpu'), temp_label.argmax(dim=1))  # 注意需要将标签从独热编码转换为索引
                loss += l1_regularization(model).to('cpu')
                # loss = 1 - dice_coeff(F.softmax(outputs.to('cpu'), dim=3), temp_label)
                total_loss += loss.item()
                total_loss += l1_regularization(model)
                dice_score += dice_coeff(F.softmax(outputs.to('cpu'), dim=1), temp_label)

                loss.backward()
                optimizer.step()

        # 测试
        test_total_loss = 0
        test_dice_score = 0
        n = 0
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_sample, batch_label in test_loader:
                batch_sample = torch.nn.functional.pad(batch_sample, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                                       mode='replicate').to('cpu')
                batch_label = torch.nn.functional.pad(batch_label, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                                      mode='replicate').to('cpu')

                loss = 0
                for i in range(iter_num_h):
                    for j in range(iter_num_w):
                        model.to(device)
                        offset_h, offset_w = i * base_h, j * base_w
                        temp_sample = batch_sample[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                        temp_label = batch_label[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]

                        outputs = model(temp_sample.to(device))
                        temp_sample.to('cpu')
                        # 计算损失
                        loss = criterion(outputs.to('cpu'), temp_label.argmax(dim=1))  # 注意需要将标签从独热编码转换为索引
                        # loss = 1 - dice_coeff(F.softmax(outputs.to('cpu'), dim=3), temp_label)
                        test_total_loss += loss.item()
                        test_dice_score += dice_coeff(F.softmax(outputs.to('cpu'), dim=1), temp_label)

        # 输出损失
    print(f'Epoch[{epoch + 1}/{num_epochs}]: Loss:{format(total_loss / len(train_loader), ".3f")}, '
          f'dice_score:{format(dice_score / (iter_num_h * iter_num_w * len(train_loader)), ".3f")}, '
          f'test_loss:{format(test_total_loss / len(test_loader), ".3f")}, '
          f'test_dice_score:{format(test_dice_score / (iter_num_h * iter_num_w * len(test_loader)), ".3f")}')

    if (test_dice_score / (iter_num_h * iter_num_w * len(test_loader))) >= best_dice:
        best_dice = (test_dice_score / (iter_num_h * iter_num_w * len(test_loader)))
        torch.save(model.state_dict(), 'temp.pt')
        print('当前最高dice：', best_dice)

#        # 调整学习率
#        if best_dice >= 0.72:
#            optimizer = optim.Adam(model.parameters(), lr=0.00002)
#        if best_dice >= 0.80:
#            optimizer = optim.Adam(model.parameters(), lr=0.00001)
#        if best_dice >= 0.85:
#            optimizer = optim.Adam(model.parameters(), lr=0.000003)
