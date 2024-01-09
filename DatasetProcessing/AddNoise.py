import random

from random import randint
# 使用数据增广的方法： 旋转 平移 噪声 模糊
import cv2

choice = random.randint(0,3)
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops
import os
import numpy as np

def color(img):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(5, 15) / 10.  # 随机因子
    # 增强因子为0.0将产生黑白图像；为1.0将给出原始图像。
    color_image = ImageEnhance.Color(img).enhance(random_factor)

    # 调整图像的饱和度
    # 增强因子为0.0将产生黑色图像；为1.0将保持原始图像
    random_factor = np.random.randint(8, 15) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)   # 调整图像的亮度

    # 增强因子为0.0将产生纯灰色图像；为1.0将保持原始图像。
    random_factor = np.random.randint(8, 13) / 10. # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

    # 增强因子为0.0将产生模糊图像；为1.0将保持原始图像，为2.0将产生锐化过的图像。
    random_factor = np.random.randint(5, 31) / 10.  # 随机因子
    random_color = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)     # 调整图像锐度
    random_color.show()
    return random_color
def move(img): #平移，平移尺度为off
    # x = np.random.randint(-1, 1)
    w, h = img.size
    print(w,h)
    offs = int((w-h)/4)
    y = np.random.randint(-offs, offs)
    offset = ImageChops.offset(img, 0, y)

    print('偏移：',y)
    offset.show()
    return offset

def gaussian_noise(image, mean=0, sigma=0.03):
    """
    添加高斯噪声
    :param image:原图
    :param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    cv2.imshow('1',output)
    cv2.waitKey(0)
    return output

img = cv2.imread('P2045.jpg')
x = gaussian_noise(img)