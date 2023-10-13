import json

import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# 示例灰度图
gray_image = np.array([
    [0, 1, 0],
    [2, 0, 1],
    [0, 0, 2],
], dtype=np.uint8)

# 假设类别映射字典，将整数值映射到标签名称
class_mapping = {
    0: "background",
    1: "tea",
    2: "impurity",
    # 添加更多的类别映射
}


# 将灰度图转换为 LabelMe 格式
def gray_to_labelme(gray_image, class_mapping):
    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": "your_image_filename.jpg",  # 请替换成实际的图像文件名
        "imageData": None,
    }

    height, width = gray_image.shape
    for label_value, label_name in class_mapping.items():
        if label_name == "background":
            continue

        mask = (gray_image == label_value).astype(np.uint8)
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

    return labelme_data


def labelme_to_gray(labelme_json_root):
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
        if shape['label'] == '2' or shape['label'] == 'impurity':
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

# 示例转换
labelme_data = gray_to_labelme(gray_image, class_mapping)

# 将 LabelMe 数据保存为 JSON 文件
with open("../output_labelme.json", "w") as json_file:
    json.dump(labelme_data, json_file, indent=2)
