# 2023.5.4
# 本程序针对多光谱演示程序修改
# 如标签数量和顺序改变，需要对代码中标签及其索引部分进行修改
# 2023.4.28添加头发检测功能，添加杂质定位和标记功能
# 2023.5.4在杂质定位标注结果图中添加比例尺
# 2023.6.16更新：
#   使用带有倾斜角度的最小矩形框标定目标位置（之前是无倾角的正矩形框），并将矩形长边的倾斜角度定义为目标整体的倾斜角度；
#   使返回的“夹点”坐标在目标区域上（之前返回的是矩形中心点坐标，不一定在目标区域上），定义“夹点”为矩形长边的中心垂直线与目标区域的交点。
# 2023.6.19更新：
#   使掩码图中茶叶(tea)的标签为1(对应背景), 减少茶叶区域对杂质检测和剔除的影响;
#   修改vis_position和vis_positions, 用带旋转角度的最小面积矩形框标注杂质位置和方向
# 2023.6.20更新：
#   detection_analyze(self, raw_img, mask)中基于掩码图mask而不是raw_img做分割，这样可以避免茶叶和杂质粘连导致判定为同一对象的情况
# 2023.8.3更新：
#   不区分具体杂质类别；去掉头发丝检测算法；
# 2023.8.17更新：
#   优化杂质的定位和姿态估计，使坐标点落在杂质上，获取坐标点附近区域的最小外接矩形框
#   考虑轮廓层级筛选
#   简化接口架构
# 2023.9.5更新：
#   针对RGB+MSI双相机系统
#   更新后处理逻辑，使从重叠样品中提取出杂质
#   推理前基于RGB图像做阈值分割，解决细小杂质检测问题
# 2023.10.17更新：
#   使用深度学习模型
#   将函数detector.vis_positions中的参数obj_contours改为mask


import os
import math
from time import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
import onnxruntime
from scipy.spatial.distance import cdist

from utils.operation import YOLO


class ImpuritiesDetector:
    def __init__(self, model_path='onnx_files', label_names_path='label_names.txt'):
        # 图片中正常显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 读取任务涉及到的所有类别名
        self.label_names = []
        with open(label_names_path, "r") as f:
            self.label_names = f.readlines()
        for i in range(len(self.label_names)):
            self.label_names[i] = self.label_names[i].strip("\n")
        # print(self.label_names)

        self.label_names2 = ['Background', 'Tea', 'Impurities']
        self.label_CN = {'Background': '非杂质区域', 'Tea': '茶叶', 'Impurities': '杂质'}

        # 记录各类杂质的索引
        self.impurity_index = [i for i, v in enumerate(self.label_names)
                               if v not in ['Background', 'Tea']]
        self.label_color = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}
        self.label_color_ = {}
        for key, value in zip(self.label_color.keys(), self.label_color.values()):
            self.label_color_[key] = [x / 255.0 for x in value]
        # 导入模型
        self.unet = onnxruntime.InferenceSession(os.path.join(model_path, 'unet.onnx'))
        self.yolo = YOLO(onnx_path=os.path.join(model_path, 'yolo.onnx'))
        self.save_dir_name = 'impurities_detect'

        # 图片尺寸信息,用于调整输入模型的子图片大小
        self.unet_base = 16
        self.init_img_size = None
        self.base_h = 656 * 2
        self.base_w = 848 * 2
        self.num_classes = 3

    def combine_img(self, folder_path):
        # 获取文件夹中的所有图像文件名
        image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]
        # 加载灰度图像并添加到列表中
        image_files.sort()
        image_list = []
        for img_path in image_files:
            img = Image.open(os.path.join(folder_path, img_path)).convert("L")  # 将图像转换为灰度模式
            image_list.append(img)

        # 确定图像的尺寸（假设所有图像都有相同的尺寸）
        width, height = image_list[0].size

        # 创建一个空的PyTorch张量，用于存储多通道图像
        multi_channel_image = np.zeros((len(image_list), height, width), dtype=np.float32)

        # 将灰度图像的像素数据叠加到PyTorch张量中
        for i, img in enumerate(image_list):
            # 将PIL图像转换为PyTorch张量
            img_tensor = transforms.ToTensor()(img)
            # 仅使用灰度通道数据
            multi_channel_image[i] = img_tensor[0]

        return multi_channel_image

    def read_data(self, path):
        """
        读取十三通道图片，生成tensor
        :param img_root: 路径应该在通道图片所在文件夹的上一层目录中
        :return:msi_arr [h, w, 13]
        """
        image = self.combine_img(path)
        self.init_img_size = image.shape[1], image.shape[2]
        # msi_arr = torch.unsqueeze(image, 0)
        # print(msi_arr.shape)

        # 将通道维度转到最后
        msi_arr = np.transpose(image, (1, 2, 0))
        return msi_arr

    def model_predict(self, data):
        # # 导入数据
        # data = self.read_data(data_path)
        # 选取第1个波长的图像作为原图显示(对比度最高)
        # raw_img = data[:, :, 0]
        # 调用模型进行检测
        data = np.transpose(data, (2, 0, 1))/255.
        height, width = self.init_img_size
        iter_num_h, iter_num_w = math.ceil(height / self.base_h), math.ceil(width / self.base_w)
        boundary_pixel_num_h, boundary_pixel_num_w = height % self.base_h, width % self.base_w
        padding_pixel_num_h, padding_pixel_num_w = (self.base_h - boundary_pixel_num_h) % self.base_h, (
                self.base_w - boundary_pixel_num_w) % self.base_w
        data = np.pad(data, ((0, 0), (0, padding_pixel_num_h), (0, padding_pixel_num_w) ), 'edge')
        image = np.expand_dims(data, axis=0)

        outputs = np.zeros(
            [image.shape[0], self.num_classes, height + padding_pixel_num_h, width + padding_pixel_num_w])

        for i in range(iter_num_h):
            for j in range(iter_num_w):
                offset_h, offset_w = i * self.base_h, j * self.base_w
                temp_sample = image[:, :, offset_h:self.base_h + offset_h, offset_w:self.base_w + offset_w]
                # temp_sample.requires_grad_(False)
                ort_inputs = {'input': temp_sample}
                output = self.unet.run(['output'], ort_inputs)[0]

                outputs[:, :self.num_classes, offset_h:self.base_h + offset_h,
                offset_w:self.base_w + offset_w] = output

        outputs = np.squeeze(outputs, 0)
        outputs = outputs[:, :self.init_img_size[0], :self.init_img_size[1]]

        mask = np.argmax(outputs, axis=0)

        return mask

    def detect(self, data):
        """
            输入待测数据，返回tuned_mask、每个杂质的定位信息
            输入：
                - data: 待测数据
            输出：
                - mask: 最终的检测掩码图
                - positions: 杂质的位置信息的列表[(R, X, Y, W), ……], shape=(n,4), n为杂质个数。以左上角为原点，横轴为x，纵轴为y
        """
        obj_contours = []
        pred_mask = self.model_predict(data)
        pred_mask[pred_mask == 1] = 0
        pred_mask[pred_mask == 2] = 1
        mask = pred_mask

        data = data[:, :, :3]

        # data = np.transpose(data, (2, 0, 1))
        data = data[:, :, ::-1] * 255.0

        det_obj = self.yolo.decect(data)

        # pred_mask = cv.cvtColor(pred_mask, cv.COLOR_GRAY2GRAY)
        for i in range(len(det_obj)):
            p = det_obj[i]['crop']
            x, y = int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2)

            pred_mask[y-20:y+20, x-20:x+20] =1

        positions = self.calculate_center(pred_mask)
        mask = np.zeros([pred_mask.shape[0], pred_mask.shape[1], 3])
        mask[pred_mask == 1] = [255, 0, 0]
        return mask, positions, obj_contours

    def calculate_center(self, gray_img, perimeter_thred=80, kernel_size=1):
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
        gray_img = cv.dilate(gray_img.astype(np.uint8), kernel, iterations=1)

        # 使用连通组件标记来标记和提取连通域
        _, labels, stats, centroids = cv.connectedComponentsWithStats(gray_img)

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
            perimeter = self.calculate_perimeter(labels, label)

            if perimeter >= perimeter_thred:
                if labels[cY, cX] <= 0:
                    # 计算中心点到该连通域内所有点的距离
                    distances = cdist(np.array([(cY, cX)]), pixels_in_component)

                    # 找到离中心点最近的点
                    min_distance_index = np.argmin(distances)
                    closest_point = tuple(pixels_in_component[min_distance_index])
                    (cY, cX) = closest_point
                connected_components_centers.append((0, cX, cY, 0))
                perimeters.append(perimeter)

        # # 将连通域和中心点绘制到图像上
        # labeled_image = gray_img

        for center_point in connected_components_centers:
            # 绘制中心点
            cv.circle(gray_img, center_point[1:3], 10, 4, -1)
            # 绘制周长
            # cv.putText(gray_img, str(int(perimeter)), center_point, cv.FONT_HERSHEY_SIMPLEX, 2, 5, 3)
            # 绘制坐标
            # cv.putText(gray_img, str(center_point), center_point, cv.FONT_HERSHEY_SIMPLEX, 2, 5, 3)

        # 坐标反转，由(宽、高)，变为(高、宽)
        # connected_components_centers = [i[::-1] for i in connected_components_centers]

        # print("后处理时间：", time() - t)
        # 显示图像
        # gray_img[gray_img == 1] = 10
        # plt.imshow(gray_img)
        # plt.axis('off')
        # plt.show()

        return connected_components_centers

    def calculate_perimeter(self, label_image, label):
        """
        计算标签为label的连通域的周长
        :param label_image: 使用cv.connectedComponentsWithStats计算出来的连通域图
        :param label:
        :return:所有标签为label的连通域周长之和
        """
        mask = (label_image == label).astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_length = 0

        for contour in contours:
            contour_length += cv.arcLength(contour, closed=True)

        return contour_length

    def draw_img(self, data, mask, save_result=False, save_path=None):
        raw_img = data[:, :, :3]

        # if save_result and save_path is None:
        #     raise Exception("请提供检测结果的保存路径 save_path")
        #
        # # 如果保存结果到本地，则在给定保存路径下创建impurities_detect文件夹
        # if save_result:
        #     save_path = os.path.join(save_path, self.save_dir_name)
        #     if not os.path.exists(save_path):
        #         os.mkdir(save_path)
        #     # 首先将原图raw_img保存到本地
        #     # cv.imwrite(os.path.join(save_path, 'raw.png'), raw_img)
        #     cv.imencode('.png', raw_img)[1].tofile(os.path.join(save_path, 'raw.png'))

        res_img = mask
        plt.imshow(res_img)
        plt.axis('off')
        plt.show()

        return res_img

    def vis_positions(self, data, positions, mask, save_result=False, save_path=None):
        # if save_result and save_path is None:
        #     raise Exception("请提供检测结果的保存路径 save_path")
        #
        # if save_result:
        #     save_path = os.path.join(save_path, self.save_dir_name)
        #     if not os.path.exists(save_path):
        #         os.mkdir(save_path)

        vis_img = data[:, :, :3]
        vis_img[:, :, 0][mask[:, :, 0] == 255] = 0
        vis_img[:, :, 1][mask[:, :, 0] == 255] = 255
        vis_img[:, :, 2][mask[:, :, 0] == 255] = 0

        plt.figure(dpi=200)
        plt.imshow(vis_img)
        plt.show()

        # plt.tight_layout()
        # if save_result:
        #     plt.savefig(os.path.join(save_path, 'positions.png'), bbox_inches='tight')
        # else:
        #     plt.show()

        return vis_img

    def save_positions(self, positions, save_path):
        # save_path = os.path.join(save_path, 'impurities_detect')
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        txt_path = os.path.join(save_path, self.save_dir_name, 'detection_res.txt')
        with open(txt_path, 'w') as f:
            f.write(f'R\tX\tY\tW\n')
            for position in positions:
                R, X, Y, W = position
                f.write(f'{R}\t{X}\t{Y}\t{W}\n')


if __name__ == '__main__':
    print(onnxruntime.get_device())
    model_path = r'onnx_files'
    label_names_path = r'label_names.txt'
    detector = ImpuritiesDetector(model_path, label_names_path)

    # 多通道数据所在路径
    test_data = os.path.join('data', '11-mix1', 'combined_data')
    data = detector.read_data(test_data)
    # positions是坐标信息
    mask, positions, obj_contours = detector.detect(data)

    # 返回掩码图
    res_img = detector.draw_img(data, mask, save_result=True, save_path=None)
    vis_img = detector.vis_positions(data, positions, mask,
                                         save_result=True,
                                         save_path=None)







    # dir_path = r'D:\mycodes\RITH\puer\data_20230921\test'
    # data_names = os.listdir(dir_path)
    # for data_name in data_names:
    #     data_path = os.path.join(dir_path, data_name, 'combined_data')
    #     print(data_path)
    #     t0 = time()
    #     data = detector.read_data(data_path)
    #     mask, positions, obj_contours = detector.detect(data)
    #     t = time() - t0
    #     print(f"检测耗时：{t}s")
    #
    #     """res_img返回掩码图"""
    #     res_img = detector.draw_img(data, mask, save_result=True, save_path=os.path.join(dir_path, data_name))
    #
    #     vis_img = detector.vis_positions(data, positions, mask,
    #                                      save_result=True,
    #                                      save_path=os.path.join(dir_path, data_name))
