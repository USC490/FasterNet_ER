import random

from random import randint
# 使用数据增广的方法： 旋转 平移 噪声 模糊
import cv2
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageChops
import os
import numpy as np
choice = random.randint(0,3)
# print(choice)
# 本代码主要提供一些针对图像分类的数据增强方法

# 1、平移。在图像平面上对图像以一定方式进行平移。
# 2、翻转图像。沿着水平或者垂直方向翻转图像。
# 3、旋转角度。随机旋转图像一定角度; 改变图像内容的朝向。
# 4、随机颜色。包括调整图像饱和度、亮度、对比度、锐度
# 5、缩放变形图片。
# 6、二值化图像。
# 7、随机黑色块遮挡
# 8、添加噪声

# 1、图像平移
def move(img): #平移，平移尺度为off
    w, h = img.size
    offs = int((w - h) / 6)
    y = np.random.randint(-offs, offs)
    offset = ImageChops.offset(img,0, y)

    print('偏移：',y)
    return offset

#  2、旋转角度
def rotation(img):
    factor = np.random.randint(-10, 10) #随机旋转角度
    # print(factor)
    rotation_img = img.rotate(factor)

    return rotation_img

# 3、随机颜色
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
    return random_color

# 4、缩放变形图片
def crop(img):
    factor_1 = np.random.randint(15, 20)
    # factor_2 = np.random.randint(14, 15)
    crop_img = img.crop((img.size[0]/factor_1, img.size[1]/factor_1, img.size[0]*(factor_1-1)/factor_1, img.size[1]*(factor_1-1)/factor_1))
    # print((img.size[0]/factor_1, img.size[1]/factor_2, img.size[0]*(factor_1-1)/factor_1, img.size[1]*(factor_2-1)/factor_2))
    cropResize_img = crop_img.resize((img.size[0], img.size[1]))
    return cropResize_img


# 5、随机添加黑白噪声
# proportion = 0.00025
def salt_and_pepper_noise(img, proportion = 0.0002):
    noise_img = img
    height,width =noise_img.size[0],noise_img.size[1]
    proportion = proportion * np.random.randint(1, 50)
    num = int(height * width * proportion) #多少个像素点添加椒盐噪声
    pixels = noise_img.load()
    for i in range(num):
        w = np.random.randint(0,width-1)
        w1 = np.random.randint(0,width-1)
        h = np.random.randint(0,height-1)
        h1 = np.random.randint(0, height - 1)
        if np.random.randint(0,2) == 1:
            pixels[h,w] = 0
        else:
            pixels[h,w] = 255
    return noise_img

# 6.添加高斯噪声
def gaussian_noise(image, mean=0, sigma=0.01):
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
    return output

def main():

    # 原始图片基本路径
    base_path = 'E:\\my_Fasternet\\crop\\upright'

    # 保存图片路径
    save_path = 'E:\\my_Fasternet\\aftercrop\\upright'
    # 增强图片文件目录

    img_num = os.listdir(base_path)
    print(img_num)
    for img_path in img_num:

        imgpath = img_path

        if img_path == 'desktop.ini':
            print('有错误')
            os.remove(base_path+'/'+'desktop.ini')
            break
        # 如果概率大于0.5 进行数据增强
        rand_rate = random.random()
        img0 = Image.open(os.path.join(base_path, imgpath))
        w, h = img0.size


        blank = (w / 2 - h) / 2
        print('blank:', blank)
        img0 = img0.crop((0, -blank, w, h + blank))

        # rand_rate 增强概率
        if rand_rate >= 0.5:
            fun_list = [random.randint(0, 6) for i in range(2)]
            # img0 = Image.open(os.path.join(base_path, imgpath))
            for func in fun_list:
                if func == 0:
                    img0= move(img0)
                    # img0.save(os.path.join(save_path, imgpath[1]))
                    # img1.save(os.path.join(save_path, imgpath[2]))

                elif func == 1:
                    img0= rotation(img0)
                    # img0.save(os.path.join(save_path, imgpath[1]))
                    # img1.save(os.path.join(save_path, imgpath[2]))

                elif func == 2:
                    img0= color(img0)
                    # img0.save(os.path.join(save_path, imgpath[1]))
                    # img1.save(os.path.join(save_path, imgpath[2]))
                elif func == 3:
                    img0= crop(img0)
                    # img0.save(os.path.join(save_path, imgpath[1]))
                    # img1.save(os.path.join(save_path, imgpath[2]))
                elif func == 4:
                    img0= salt_and_pepper_noise(img0,proportion=0.0003)
                    # img0.save(os.path.join(save_path, imgpath[1]))
                    # img1.save(os.path.join(save_path, imgpath[2]))
                elif func == 5:
                    img0 = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
                    img0=gaussian_noise(img0, mean=0, sigma=0.0009)
                    img0 = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
                else:
                    print('fun:',func)
                    print('未执行图片变换函数')
                    continue
            img0.save(os.path.join(save_path, imgpath))
        else:

            # 保存图片
            img0.save(os.path.join(save_path, imgpath))



if __name__ == "__main__":
    main()
