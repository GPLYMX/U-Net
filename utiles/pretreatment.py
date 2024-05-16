import os
import re
import math

import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage import measure


def get_size(size, base=16):
    """
    使得图片的尺寸为base的整数倍
    size为原始尺寸，格式是列表譬如[120, 300]
    输出格式为列表
    """
    base = float(base)
    rate1 = math.ceil(size[0] / base)
    rate2 = math.ceil(size[1] / base)
    return (int(rate1 * base), int(rate2 * base))


def resize_img(root=r'F:\GK\pictures\output1692444903.png'):
    """
    读取并返回修改好尺寸的图片(长*高*通道数)
    """
    img = cv2.imread(root)
    print(img.shape)
    if img.any():
        size = get_size([img.shape[1], img.shape[0]])
        img = cv2.resize(img, size)
        return img
    else:
        pass
        # print('图片读取失败')


def get_label(filename):
    """使用正则表达式获取数据标签"""
    try:
        label = re.search(r'-(\d+)', filename).group(1)
        return int(label)
    except:
        print('匹配失败')
        return None


def kmean(data):
    """使用kmean算法使得背景图为0，杂质和茶叶为1"""
    reshaped_data = data.reshape((-1, data.shape[2]))
    # 设置聚类的簇数
    num_clusters = 2

    # 通过设置初始聚类中心，确保向量绝对值更大的类别被标记为0类
    initial_centers = np.array([np.mean(reshaped_data, axis=0), np.zeros(13)])

    # 创建KMeans模型并进行聚类
    kmeans = KMeans(n_clusters=num_clusters, init=initial_centers, n_init=1, random_state=0)
    # 获取聚类标签
    cluster_labels = kmeans.fit(reshaped_data).labels_
    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_

    # 将聚类结果重新恢复为图像的形状
    clustered_image = cluster_labels.reshape(data.shape[0], data.shape[1])

    # # 使用连通域分析对1类的像素进行处理
    # threshold = 10000
    # label_image = measure.label(clustered_image, connectivity=2)
    # for region in measure.regionprops(label_image):
    #     if region.label == 1:  # 1类的连通域
    #         if region.area < threshold:  # 当连通域小于阈值时，标记为0类
    #             clustered_image[label_image == region.label] = 0

    # 标签1的像素点转为255
    clustered_image = (clustered_image * 255 / (num_clusters - 1)).astype(np.uint8)

    return clustered_image


def get_cluster_img(filename):
    label = get_label(filename)

    if label == None:
        print(label)
        return None, None, None

    cwl_path = os.path.join(data_path, filename, 'combined_data')
    gray_images = []
    for file in os.listdir(cwl_path):
        image = cv2.imread(os.path.join(cwl_path, file), cv2.IMREAD_GRAYSCALE)
        # plt.imshow(image)
        gray_images.append(image)
    thirteen_channel_array = cv2.merge(gray_images)
    img = kmean(thirteen_channel_array)

    if int(label) != 1:
        img[img == 255] = int(2)
    else:
        img[img == 255] = int(1)

    print("类别为：", label)
    plt.imshow(img)
    plt.show()
    # print(img.max())
    # print(img.shape)
    cv2.imwrite(os.path.join(data_path, filename, filename + 'label.png'), img)

    return thirteen_channel_array, img, label


def rename_file(folder_path=r"D:\MyCodes\RITH\puer\data_20230901\data\train\1-0\combined_data"):
    """
    将1、2、3文件命名为01、02、03
    :param folder_path:
    :return:
    """
    # 获取文件夹中的所有图像文件名
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]

    # 对文件名进行排序，确保按正确的顺序加载图像
    image_files.sort()

    # 计算需要的宽度以包含所有文件的编号
    width = len(str(len(image_files)))

    # 重命名文件
    for i, img_file in enumerate(image_files, start=1):
        new_file_name = f"{i:0{width}d}.png"
        old_file_path = os.path.join(folder_path, img_file)
        new_file_path = os.path.join(folder_path, new_file_name)
        try:
            os.rename(old_file_path, new_file_path)
        except FileExistsError:
            pass

    print("文件名已重新命名。")


if __name__ == '__main__':
    data_path = r'D:\MyCodes\RITH\puer\data_20230901\data\train'

    """
    批量修改图片命名格式
    """
    for file in os.listdir(data_path):
        rename_file(os.path.join(data_path, file, 'combined_data'))

    """
    通过聚类的方式获取标签，并将标签保存到训练集对应的文件夹里。并显示聚类效果
    """
    datas = []
    for filename in os.listdir(data_path):
        merge_img, label_img, label = get_cluster_img(filename)
        if merge_img is None:
            continue
        datas.append([merge_img, label_img, label])
