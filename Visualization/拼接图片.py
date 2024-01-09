import PIL.Image as Image
from PIL import ImageDraw, ImageFont
from PIL import ImageChops
import os

IMAGES_PATH = 'C:\\Users\\Admin\\Desktop\\pic\\te\\'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG','.png']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'C:\\Users\\Admin\\Desktop\\final.jpg'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]


#排序，这里需要根据自己的图片名称切割，得到数字
image_names.sort(key=lambda x:int(x.split(("."),2)[0]))
print(image_names)
#image_names.sort(key=lambda x:int(x[:-4]))

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")

padding=5
head_padding=50
# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB',(IMAGE_COLUMN * IMAGE_SIZE+padding*(IMAGE_COLUMN-1), head_padding+IMAGE_ROW * IMAGE_SIZE+padding*(IMAGE_ROW-1)-140),'white' )  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            if y==1:
                to_image.paste(from_image, ((x - 1) * IMAGE_SIZE+padding* (x - 1), head_padding+(y - 1) * IMAGE_SIZE+padding* (y - 1)))
            else:
                to_image.paste(from_image, (
                (x - 1) * IMAGE_SIZE + padding * (x - 1), head_padding + (y - 1) * IMAGE_SIZE + padding * (y - 1) - 80))
    w, h = to_image.size
    offs = int((w - h) / 10)
    to_image = ImageChops.offset(to_image, 0, -offs)
    draw = ImageDraw.Draw(to_image)
    text = ['a.close \n 闭眼','b.up \n向上','c.down \n 向下','d.left \n 向左','e.right \n 向右']
    text1 = ['f.forward \n  正视','g.leftup \n 向左上','h.leftdown \n  向左下','i.rightup \n  向右上','j.rightdown \n  向右下']
    text_offset = [70,100,90,90,90]
    text_offset1 = [60,70,70,70,70]
    # font = ImageFont.truetype(r'C:\Windows\Fonts\timesbd.ttf', 32)

    font = ImageFont.truetype("simhei.ttf",32,encoding="utf-8")

    for i in range(1, IMAGE_COLUMN + 1):
        draw.text((text_offset[i-1]+256*(i-1), 150), text[i-1], fill='#666', font=font)
        draw.text((text_offset1[i - 1] + 256 * (i - 1), 330), text1[i - 1], fill='#666', font=font)

    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


image_compose()  # 调用函数
