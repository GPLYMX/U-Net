import math
import time

import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.network import AttU_Net, U_Net
from utiles.utiles import CustomSegmentationDataset, visualization

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('use_GPU:', True)
    device = torch.device('cuda')
else:
    print('use_GPU:', False)
    device = torch.device('cpu')


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


configs = load_configs('configs.yaml')
num_classes = configs['num_classes']
in_channels = configs['in_channels']
batch_size = configs['batch_size']
num_epochs = configs['num_epochs']

for i in range(10):
    torch.cuda.empty_cache()

t1 = time.time()
# 定义损失函数和优化器
model = AttU_Net(img_ch=in_channels, output_ch=num_classes)
model.load_state_dict(torch.load('temp.pt'))
# model.eval()
dataset = CustomSegmentationDataset(data_dir=r'D:\mycodes\RITH\puer\data_20230921\test', mode='test',
                                    num_channel=in_channels)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = model.to(device)
t2 = time.time()

n = 0
for batch_sample, img_name in test_loader:
    n = n + 1
    batch_sample = batch_sample.requires_grad_(False)

    base_h, base_w = configs['base_size']

    height, width = batch_sample.shape[2], batch_sample.shape[3]
    iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
    boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
    padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
                base_w - boundary_pixel_num_w) % base_w

    batch_sample = torch.nn.functional.pad(batch_sample, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                           mode='replicate').to('cpu')
    outputs = torch.zeros_like(batch_sample)[:, 0:num_classes, :, :]

    loss = 0
    with torch.no_grad():
        for i in range(iter_num_h):
            for j in range(iter_num_w):
                model.to(device)
                offset_h, offset_w = i * base_h, j * base_w
                temp_sample = batch_sample[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                temp_sample = temp_sample.to(device)
                # model.half()
                # temp_sample = temp_sample.half()
                output = model(temp_sample).to('cpu')

                outputs[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w] = output
                temp_sample.to('cpu')
    outputs = outputs[:, :, 0:height, 0:width]



    # h = math.ceil(batch_sample.shape[2] / 2)
    # w = math.ceil(batch_sample.shape[3] / 2)
    # batch_list = [batch_sample[:, :, h:, :w], batch_sample[:, :, h:, w:], batch_sample[:, :, :h, :w],
    #               batch_sample[:, :, :h, w:]]
    #
    # outputs = []
    # with torch.no_grad():
    #     for i in range(4):
    #         model = model.to(device)
    #         batch_list[i] = batch_list[i].requires_grad_(False)
    #         outputs.append(model(batch_list[i].to(device)).to('cpu'))
    #         model = model.to('cpu')
    #         torch.cuda.empty_cache()
    # outputs = torch.cat([torch.cat([outputs[2], outputs[3]], dim=3), torch.cat([outputs[0], outputs[1]], dim=3)], dim=2)
    for i in range(outputs.shape[0]):
        outputs = visualization(outputs[i])
        plt.imshow(outputs)
        plt.title(img_name)
        plt.axis('off')
        plt.show()

t3 = time.time()
print('模型加载时间：', t2 - t1)
print('单张图片预测时间：', (t3 - t2) / n)
