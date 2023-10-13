import math
import os
import random
import json
import yaml

from PIL import Image, ImageDraw
import torch
import cv2
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
configs = load_configs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs.yaml'))
unet_base = configs['unet_base']


def get_size(size, base=unet_base):
    """
    使得图片的尺寸为base的整数倍
    size为原始尺寸，格式是列表譬如[120, 300]
    输出格式为列表
    """
    base = float(base)
    rate1 = math.ceil(size[0] / base)
    rate2 = math.ceil(size[1] / base)
    return (int(rate1 * base), int(rate2 * base))


def resize_img(img, base=unet_base):
    """
    读取并返回修改好尺寸的图片
    base: unet下采样的次数为n，则base为2^n
    """
    try:
        if isinstance(img, Image.Image):
            width, height = img.size
            size = get_size([width, height], base=base)
            img = img.resize(size)
        if isinstance(img, torch.Tensor):
            # 获取原始张量的高度和宽度
            height, width = img.shape[1], img.shape[2]

            # # 计算需要添加的垂直和水平填充量
            # padding_height = (base - (height % base)) % base
            # padding_width = (base - (width % base)) % base

            #
            # # 使用 F.pad 函数添加填充，将高度和宽度调整为16的整数倍
            # img = torch.nn.functional.pad(img, (0, padding_width, 0, padding_height), mode='constant', value=189)

            # 计算要添加的填充量
            H_padding = (base - (height % base)) % base
            W_padding = (base - (width % base)) % base

            # 使用 'replicate' 模式进行填充
            img = torch.nn.functional.pad(img, (0, W_padding, 0, H_padding), mode='replicate')

        return img
    except Exception as e:
        print(e)
        print('图片读取失败')


# 将灰度图转换为 LabelMe 格式
def gray_to_labelme(gray_image, class_mapping={0: "background", 1: "tea", 2: "impurity", }):

    #     gray_image = cv2.imread(gray_image_root)
    #     gray_image = np.array(gray_image)

    height, width = gray_image.shape
    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": gray_image,  # 请替换成实际的图像文件名
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    for label_value, label_name in class_mapping.items():
        if label_name == "background":
            continue

        mask = (gray_image == label_value).astype(np.uint8)
        print(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            points = contour.squeeze().tolist()
            shape_data = {
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            labelme_data["shapes"].append(shape_data)

    # 将数据保存为JSON文件
    #     with open('labelme_data.json', 'w') as json_file:
    #         json.dump(labelme_data, json_file)

    return labelme_data


class CustomSegmentationDataset(Dataset):

    def __init__(self, data_dir, transform=None, num_classes=3, mode='train', augment=True, channels=[0, 1, 2],
                 label_format='json'):
        """
        :param data_dir: 图片所在路径，一般是combined_data的上一级目录
        :param transform:
        :param num_classes:
        :param mode: mode=train时，__getitem__会返回处理后的训练图像和标签，mode=test时返回训练图片和文件名
        :param augment: 是否用图片拼接的方式做数据增强
        :param num_channel:输入的通道数
        :param label_format:代表标签的格式，标签是json文件时名字为'00.json'，放在combined_data文件夹下；标签是png时，名字是’label.png'，放在combined_data的文件夹的上一级目录下
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = os.listdir(self.data_dir)
        self.num_classes = num_classes
        self.mode = mode
        self.augment = augment
        self.channels = channels
        self.label_format = label_format

    def __len__(self):
        return len(self.file_names)

    def read_gray_label(self, label):
        """
        读取label，并转化为独热编码格式的图片
        :param label_root:
        :return:
        """
        # label = Image.open(label_root).convert('L')
        label_image = transforms.ToTensor()(label)
        label_image = resize_img(label_image) * 255.

        # if self.transform:
        #     image = self.transform(image) / 255.0
        #     label_image = self.transform(label)
        # 创建独热编码的标签
        label_image_onehot = np.zeros((self.num_classes, label_image.shape[1], label_image.shape[2]))
        for class_idx in range(self.num_classes):
            label_image_onehot[class_idx, :, :] = (label_image == class_idx)
        label_image_onehot = torch.tensor(label_image_onehot, dtype=torch.float32)  # 转换为 PyTorch 张量
        # print('a', label_image_onehot[0,:,:].max())
        # print('b', label_image_onehot[1, :, :].max())
        # print('c', label_image_onehot[2, :, :].max())
        return label_image_onehot

    def json_to_gray(self, json_root):
        """
        读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
        """
        # # 1. 解析LabelMe标注文件（JSON格式）
        with open(json_root, 'r') as json_file:
            labelme_data = json.load(json_file)

        # 2. 获取图像尺寸
        image_width = labelme_data['imageWidth']
        image_height = labelme_data['imageHeight']

        # 3. 创建灰度图
        gray_image = Image.new('L', (image_width, image_height), 0)

        # 4. 为每个对象分配类别值
        category_mapping = {}  # 用于将类别名称映射到整数值
        category_id = 1

        for shape in labelme_data['shapes']:
            print(shape)
            category_name = shape['label']
            if category_name not in category_mapping:
                category_mapping[category_name] = category_id
                category_id += 1

            category_value = category_mapping[category_name]
            if isinstance(shape['points'][0], list):
                # 创建多边形的坐标列表
                polygon_points = [(int(x), int(y)) for x, y in shape['points']]

                # 使用PIL的绘图功能填充多边形区域
                draw = ImageDraw.Draw(gray_image)
                draw.polygon(polygon_points, fill=category_value)
        # 5. 保存灰度图
        gray_image = np.array(gray_image)
        gray_image = Image.fromarray(gray_image)
        # gray_image.save('output_gray_image.png')
        return gray_image

    def read_img(self, img_root):
        img_root = os.path.join(self.data_dir, img_root, 'combined_data')
        image = self.combine_img(img_root)
        image = resize_img(image) / 255.
        return image

    def read_label(self, label_name):
        """
        读取label的文件名，返回独热编码
        :param label_name:
        :return:
        """
        if self.label_format == 'json':
            label_name = os.path.join(self.data_dir, label_name, "pretreatment", 'RGB.json')
            gray_img = self.labelme_to_gray(label_name)
            label_image_onehot = self.read_gray_label(gray_img)
        if self.label_format == 'png':
            label_name = os.path.join(self.data_dir, label_name, "pretreatment", 'label.png')
            gray_image = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
            gray_image = np.array(gray_image)
            gray_image[gray_image==3] = 1
            gray_image = Image.fromarray(gray_image)
            label_image_onehot = self.read_gray_label(gray_image)
        return label_image_onehot

    def __getitem__(self, idx):
        # 指定图像和标签的文件格式为PNG
        image = self.read_img(self.file_names[idx])

        if self.mode == 'test':
            return image, self.file_names[idx]
        try:
            label_image_onehot = self.read_label(self.file_names[idx])

            """
            不选择数据增加就直接输出，否则进行数据拼接
            """
            if not self.augment:
                return image, label_image_onehot
            else:
                random_idx = random.randint(0, self.__len__() - 1)
                image1, label_image_onehot1 = self.read_img(self.file_names[random_idx]), self.read_label(self.file_names[random_idx])

                # 随机生成一个矩形框
                h, w = image.shape[1], image.shape[2]
                h1 = random.randint(0, h - 3)
                h2 = random.randint(h1 + 1, h - 1)
                w1 = random.randint(0, w - 3)
                w2 = random.randint(w1 + 1, w - 1)

                image[:, h1:h2, w1:w2] = image1[:, h1:h2, w1:w2]
                label_image_onehot[:, h1:h2, w1:w2] = label_image_onehot1[:, h1:h2, w1:w2]

            return image, label_image_onehot

        except FileNotFoundError:
            """
            测试模式下不需要label信息，直接返回image
            """
            print(self.file_names[idx])
            return image, self.file_names[idx]

    def combine_img(self, folder_path):
        # 获取文件夹中的所有图像文件名
        image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]
        image_files = [image_files[i] for i in self.channels]
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
            img_tensor = transforms.ToTensor()(img)  # 将PIL图像转换为PyTorch张量
            multi_channel_image[i] = img_tensor[0]  # 仅使用灰度通道数据
        return multi_channel_image

    def labelme_to_gray(self, labelme_json_root):
        """
        读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
        """
        # # 1. 解析LabelMe标注文件（JSON格式）
        with open(labelme_json_root, 'r') as json_file:
            labelme_data = json.load(json_file)

        # 2. 获取图像尺寸
        image_width = labelme_data['imageWidth']
        image_height = labelme_data['imageHeight']

        # 3. 创建灰度图
        gray_image = Image.new('L', (image_width, image_height), 0)

        # 4. 为每个对象分配类别值
        category_mapping = {}  # 用于将类别名称映射到整数值
        category_id = 1

        for shape in labelme_data['shapes']:
            category_name = shape['label']
            if category_name not in category_mapping:
                category_mapping[category_name] = category_id
                category_id += 1

            category_value = 0
            if shape['label'] == '1' or shape['label'] == 'tea':
                category_value = 1
            if shape['label'] == '2' or shape['label'] == '3' or shape['label'] == 'impurity':
                category_value = 2
            if isinstance(shape['points'][0], list):
                # 创建多边形的坐标列表
                polygon_points = [(int(x), int(y)) for x, y in shape['points']]

                # 使用PIL的绘图功能填充多边形区域
                draw = ImageDraw.Draw(gray_image)
                draw.polygon(polygon_points, fill=category_value)
        # 5. 保存灰度图
        gray_image = np.array(gray_image)
        gray_image = Image.fromarray(gray_image)
        # gray_image.save('output_gray_image.png')
        return gray_image


def visualization(torch_matrix):
    """
    将模型训练出来的torch格式矩阵[1*3*h*w]转化为plt格式的图片
    :return:
    """
    torch_matrix = torch_matrix.squeeze(0)
    try:
        np_matrix = torch_matrix.detach().numpy()
    except TypeError:
        torch_matrix = torch_matrix.to('cpu')
        np_matrix = torch_matrix.detach().numpy()

    max_channel_indices = np.argmax(np_matrix, axis=0)

    # 第一个通道最大的情况，将所有值设为0
    np_matrix[0, max_channel_indices == 0] = 0
    np_matrix[1, max_channel_indices == 0] = 0
    np_matrix[2, max_channel_indices == 0] = 0
    # 第二个通道最大的情况，标记为黄色
    np_matrix[0, max_channel_indices == 1] = 255
    np_matrix[1, max_channel_indices == 1] = 255
    np_matrix[2, max_channel_indices == 1] = 0
    # 第三个通道最大的情况，标记为红色
    np_matrix[0, max_channel_indices == 2] = 255
    np_matrix[1, max_channel_indices == 2] = 0
    np_matrix[2, max_channel_indices == 2] = 0

    np_matrix = np.transpose(np_matrix, (1, 2, 0))

    return np_matrix


def dice_coeff(predicted, target, epsilon=1e-5):
    predicted = predicted.squeeze(0)
    target = target.squeeze(0)
    try:
        predicted = predicted.detach().numpy()
    except TypeError:
        predicted = predicted.to('cpu')
        np_matrix = predicted.detach().numpy()
        target = target.to('cpu')

    max_channel_indices = np.argmax(np_matrix, axis=0)

    # 第一个通道最大的情况，将所有值设为0
    np_matrix[0, max_channel_indices == 0] = 1
    np_matrix[1, max_channel_indices == 0] = 0
    np_matrix[2, max_channel_indices == 0] = 0
    # 第二个通道最大的情况，标记为黄色
    np_matrix[0, max_channel_indices == 1] = 0
    np_matrix[1, max_channel_indices == 1] = 1
    np_matrix[2, max_channel_indices == 1] = 0
    # 第三个通道最大的情况，标记为红色
    np_matrix[0, max_channel_indices == 2] = 0
    np_matrix[1, max_channel_indices == 2] = 0
    np_matrix[2, max_channel_indices == 2] = 1

    predicted = torch.tensor(np_matrix)

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) + epsilon
    dice = (2.0 * intersection) / union
    return dice
