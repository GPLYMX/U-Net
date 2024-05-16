# -*- coding: utf-8 -*-
# @Time : 2023/9/25 13:55
# @Author : GuoPeng
from time import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def calculate_center(gray_img, perimeter_thred=180, kernel_size=5):
    """
    读取一张0、1二值灰度图，剔除周长小于perimeter_thred的连通域，返回剩余连通域的中心点
    :param gray_img: 0、1二值灰度图
    :param perimeter_thred: 周长阈值
    :param kernel_size: 膨胀核大小
    :return:连通域的中心点[(高、宽)， (高、宽)·····]
    """

    # 膨胀操作，合并连通域
    t = time()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_img = cv2.dilate(gray_img.astype(np.uint8), kernel, iterations=2)

    # 使用连通组件标记来标记和提取连通域
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_img)

    # 初始化变量来存储连通域的中心点坐标和最近的点坐标
    connected_components_centers = []
    perimeters = []

    # 遍历每个连通域，跳过背景（标签为0）
    for label in range(1, len(stats)):
        # 获取连通域的中心点
        cX, cY = centroids[label]
        cX, cY = int(cX), int(cY)

        # 获取当前连通域的像素坐标
        pixels_in_component = np.argwhere(labels == label)

        # 计算周长
        perimeter = calculate_perimeter(labels, label)

        if perimeter >= perimeter_thred:
            if labels[cY, cX] <= 0:
                # 计算中心点到该连通域内所有点的距离
                distances = cdist(np.array([(cY, cX)]), pixels_in_component)

                # 找到离中心点最近的点
                min_distance_index = np.argmin(distances)
                closest_point = tuple(pixels_in_component[min_distance_index])
                (cY, cX) = closest_point
            connected_components_centers.append((cX, cY))
            perimeters.append(perimeter)

    # # 将连通域和中心点绘制到图像上
    # labeled_image = gray_img

    for center_point, perimeter in zip(connected_components_centers, perimeters):
        # 绘制中心点

        # cv2.circle(gray_img, center_point, 10, 4, -1)
        # 绘制周长
        cv2.putText(gray_img, str(int(perimeter)), center_point, cv2.FONT_HERSHEY_SIMPLEX, 2, 5, 3)
        # 绘制坐标
        # cv2.putText(gray_img, str(center_point), center_point, cv2.FONT_HERSHEY_SIMPLEX, 2, 5, 3)

    # 坐标反转，由(宽、高)，变为(高、宽)
    # connected_components_centers = [i[::-1] for i in connected_components_centers]

    print('杂质个数：', len(connected_components_centers))
    print("后处理时间：", time()-t)
    # 显示图像
    gray_img[gray_img==1] = 10
    plt.imshow(gray_img)
    plt.axis('off')
    plt.show()

    return connected_components_centers


def calculate_perimeter(label_image, label):
    """
    计算标签为label的连通域的周长
    :param label_image: 使用cv2.connectedComponentsWithStats计算出来的连通域图
    :param label:
    :return:所有标签为label的连通域周长之和
    """
    mask = (label_image == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = 0

    for contour in contours:
        contour_length += cv2.arcLength(contour, closed=True)

    return contour_length


if __name__ == '__main__':
    gray_img = cv2.imread(r'output_gray_image.png', cv2.IMREAD_GRAYSCALE)
    t1 = time()
    points = calculate_center(gray_img)
    print(time() - t1)
    print(points)
