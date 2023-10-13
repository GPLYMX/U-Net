# -*- coding: utf-8 -*-
# @Time : 2023/9/19 14:15
# @Author : GuoPeng
import os
import math
import yaml
import time

from PIL import Image
import torch
from torchvision import transforms
from matplotlib import pyplot as plt

from models.network import AttU_Net, U_Net
from utiles.postprocessing import calculate_center

"""
将十三通道的图片放入模型中预测，根据预测的结果生成灰度图，灰度图中的数字代表类别
"""
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
base_h, base_w = configs['base_size']


def combine_img(folder_path):
    # 获取文件夹中的所有图像文件名
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]
    image_files = [image_files[i] for i in in_channels]
    # 加载灰度图像并添加到列表中
    image_files.sort()
    image_list = []
    for img_path in image_files:
        img = Image.open(os.path.join(folder_path, img_path)).convert("L")  # 将图像转换为灰度模式
        image_list.append(img)

    # 确定图像的尺寸（假设所有图像都有相同的尺寸）
    width, height = image_list[0].size

    # 创建一个空的PyTorch张量，用于存储多通道图像
    multi_channel_image = torch.zeros(len(image_list), height, width)

    # 将灰度图像的像素数据叠加到PyTorch张量中
    for i, img in enumerate(image_list):
        # 将PIL图像转换为PyTorch张量
        img_tensor = transforms.ToTensor()(img)
        # 仅使用灰度通道数据
        multi_channel_image[i] = img_tensor[0]
    torch.cuda.synchronize()
    a = time.time()
    torch.cuda.synchronize()
    multi_channel_image.to(device)
    torch.cuda.synchronize()
    print('从CPU转移到cuda耗时:', time.time()-a)
    torch.cuda.synchronize()
    return multi_channel_image


def read_img(img_root):
    """
    读取十三通道图片，生成tensor
    :param img_root: 路径应该在通道图片所在文件夹的上一层目录中
    :return:model需要的格式, 原始图片的尺寸
    """
    image = combine_img(img_root) / 255.
    h, w = image.shape[1], image.shape[2]

    image = torch.unsqueeze(image, 0)

    return image, [h, w]


def model_pred(image, model, shape):
    # image = resize_img(image)
    # h = math.ceil(image.shape[2] / 2)
    # w = math.ceil(image.shape[3] / 2)
    # batch_list = [image[:, :, h:, :w], image[:, :, h:, w:], image[:, :, :h, :w],
    #               image[:, :, :h, w:]]
    #
    # outputs = []
    # with torch.no_grad():
    #     for i in range(4):
    #         model.to(device)
    #         batch_list[i] = batch_list[i].requires_grad_(False)
    #         outputs.append(model(batch_list[i].to(device)).to('cpu'))
    #
    # outputs = torch.cat([torch.cat([outputs[2], outputs[3]], dim=3), torch.cat([outputs[0], outputs[1]], dim=3)], dim=2)
    torch.cuda.synchronize()
    ttt = time.time()
    torch.cuda.synchronize()
    height, width = shape
    iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
    boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
    padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
            base_w - boundary_pixel_num_w) % base_w
    image = torch.nn.functional.pad(image, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                    mode='replicate').to(device)

    outputs = torch.zeros([image.shape[0], num_classes, height + padding_pixel_num_h, width + padding_pixel_num_w]).to(
        device)
    # outputs = image[:, :num_classes, :, :]
    print("图片分割时间:", time.time() - ttt)
    # model.eval()
    # if use_gpu:
    #     model.half()
    #     image = image.half()
    with torch.no_grad():
        for i in range(iter_num_h):
            for j in range(iter_num_w):
                offset_h, offset_w = i * base_h, j * base_w
                temp_sample = image[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                temp_sample.requires_grad_(False)
                output = model(temp_sample)

                outputs[:, :num_classes, offset_h:base_h + offset_h, offset_w:base_w + offset_w] = output

    # outputs = outputs[:, :, 0:height, 0:width]
    torch.cuda.synchronize()
    print('单张预测时间：', time.time() - ttt)
    torch.cuda.synchronize()

    return outputs


def get_impurity_mask(img_root, model):
    """
    读取十三通道图片，生成杂质的掩码图
    :param img_root:
    :return:
    """
    img_root = os.path.join(img_root, 'combined_data')
    torch.cuda.synchronize()
    t = time.time()
    torch.cuda.synchronize()
    img, shape = read_img(img_root)
    print(shape)
    torch.cuda.synchronize()
    print('预处理时间', time.time() - t)
    torch.cuda.synchronize()
    img = model_pred(img, model, shape)
    img = torch.squeeze(img, 0)
    img = img[:, :shape[0], :shape[1]]

    img = torch.argmax(img, axis=0)
    img[img == 1] = 0
    img[img == 2] = 1

    torch.cuda.synchronize()
    a = time.time()
    torch.cuda.synchronize()
    img = img.cpu()
    img = img.detach().numpy()
    torch.cuda.synchronize()
    print("数据从cuda迁移到cpu耗时", time.time()-a)
    torch.cuda.synchronize()

    return img


if __name__ == '__main__':
    t0 = time.time()
    model = AttU_Net(img_ch=len(in_channels), output_ch=num_classes)
    model.load_state_dict(torch.load('temp.pt'))
    model.to(device)
    # model.eval()
    t1 = time.time()
    print("模型加载时间：", t1 - t0)
    root = r'D:\mycodes\RITH\puer\data_20230921\test\7-mix1'
    img = get_impurity_mask(root, model)
    calculate_center(img)
    print("计算杂质掩码所需总时间", time.time() - t0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
