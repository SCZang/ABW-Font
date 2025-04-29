# -*- coding:utf-8 -*-
"""

作者：Exiler
日期：2023年11月13日
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rotate_and_compute_projections(image_path, angles=[0, 15, 30, 45, 60, 75]):

    # 打开图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 反转像素值
    inverted_img = 255 - img

    # 逐个角度旋转图像并计算投影
    for i, angle in enumerate(angles):
        rotated_img = rotate_image(inverted_img, angle)

        horizontal_projection = np.sum(rotated_img, axis=0)
        vertical_projection = np.sum(rotated_img, axis=1)

        normalized_horizontal_projection = normalize_histogram(horizontal_projection)
        normalized_vertical_projection = normalize_histogram(vertical_projection)

        # 绘制原始图像和投影图像
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(normalized_horizontal_projection)
        plt.title(f'{image_path}Rotated {angle}° Horizontal Projection')
        plt.xlabel('Column')
        plt.ylabel('Normalized Intensity')

        plt.subplot(1, 2, 2)
        plt.plot(normalized_vertical_projection)
        plt.title(f'{image_path}Rotated {angle}° Vertical Projection')
        plt.xlabel('Row')
        plt.ylabel('Normalized Intensity')

        plt.tight_layout()
        filename = os.path.splitext(os.path.basename(image_path))[0]
        plt.savefig(f'res_all/{filename}_{angle}.png')
        #plt.show()

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def normalize_histogram(hist):
    return hist / np.max(hist)

# 调用函数并传入图像路径

#rotate_and_compute_projections('img/2563.png')
# 获取图像文件夹中的所有文件
image_folder = 'img_all/'
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 遍历所有文件并调用 rotate_and_compute_projections 函数
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # 关闭所有之前创建的图形
    plt.close('all')

    rotate_and_compute_projections(image_path)
