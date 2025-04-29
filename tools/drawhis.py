# -*- coding:utf-8 -*-
"""

作者：Exiler
日期：2023年11月12日
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize_histogram(projection):
    # 归一化直方图
    normalized_projection = projection / np.sum(projection)
    return normalized_projection

# def plot_normalized_histogram(image_path):
#     # 打开图像
#     img = Image.open(image_path).convert('L')  # 转为灰度图
#
#     # 将图像转为 NumPy 数组
#     img_array = np.array(img)
#
#     # 计算水平和垂直方向的1D投影
#     horizontal_projection = np.sum(img_array, axis=0)
#     vertical_projection = np.sum(img_array, axis=1)
#
#     # 归一化直方图
#     normalized_horizontal_projection = normalize_histogram(horizontal_projection)
#     normalized_vertical_projection = normalize_histogram(vertical_projection)
#
#     # 绘制归一化直方图
#     plt.figure(figsize=(12, 6))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(normalized_horizontal_projection)
#     plt.title('Normalized Horizontal Projection')
#     plt.xlabel('Column')
#     plt.ylabel('Normalized Intensity')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(normalized_vertical_projection)
#     plt.title('Normalized Vertical Projection')
#     plt.xlabel('Row')
#     plt.ylabel('Normalized Intensity')
#
#     plt.tight_layout()
#     plt.show()


def plot_normalized_histogram(image_path):
    # 打开图像
    img = Image.open(image_path).convert('L')  # 转为灰度图

    # 将图像转为 NumPy 数组
    img_array = np.array(img)

    # 反转像素值
    inverted_img_array = 255 - img_array

    # 计算水平和垂直方向的1D投影
    horizontal_projection = np.sum(inverted_img_array, axis=0)
    vertical_projection = np.sum(inverted_img_array, axis=1)

    # 归一化直方图
    normalized_horizontal_projection = normalize_histogram(horizontal_projection)
    normalized_vertical_projection = normalize_histogram(vertical_projection)

    # 绘制归一化直方图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(normalized_horizontal_projection)
    plt.title('Normalized Horizontal Projection')
    plt.xlabel('Column')
    plt.ylabel('Normalized Intensity')

    plt.subplot(1, 2, 2)
    plt.plot(normalized_vertical_projection)
    plt.title('Normalized Vertical Projection')
    plt.xlabel('Row')
    plt.ylabel('Normalized Intensity')

    plt.tight_layout()
    plt.show()
# 调用函数并传入图像路径
plot_normalized_histogram('img/0426.png')
plot_normalized_histogram('img/0427.png')