import cv2
import os
def resize_img_keep_ratio(src, target_size):
    # img = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)  # 读取图片
    img = src.copy()
    old_size = img.shape[0:2]  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    # img_new = Image.fromarray(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
    return img_new


# 读取文件夹
def getfiles(file):
    path_list = []
    filenames = os.listdir(file)
    for filename in filenames:
        a = os.path.join(file, filename)
        # print(a)
        path_list.append(a)
    # print(path_list)
    return path_list

def main(path):
    img_list = getfiles(path)
    _, save_file = os.path.split(path)
    if not os.path.exists(save_file):
        os.mkdir(save_file)

    for img in img_list:
        image = cv2.imread(img)
        new_img = resize_img_keep_ratio(image, (224, 224))
        _,filename = os.path.split(img)

        cv2.imwrite(os.path.join(save_file,filename), new_img)




if __name__ == '__main__':
    path = 'D:\\eye_label\\right_down_look'
    main(path)
