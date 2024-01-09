from PIL import Image
def Padding_PIC(img):
    w, h = img.size

    if max(w, h) <= 600:
        w_s = w  #
        h_s = h  #
    elif max(w, h) > 800:
        w_s = int(w / 2.2)  # 长宽缩小两倍
        h_s = int(h / 2.2)  # 长宽缩小两倍

    else:
        w_s = int(w / 1.8)  # 长宽缩小两倍
        h_s = int(h / 1.8)  # 长宽缩小两倍
    img = img.resize((w_s, h_s), Image.ANTIALIAS)
    blank = (w_s - h_s) / 2

    # img.crop((w0, h0, w1, h1)) w0，h0宽度，高度起始方向剪裁的值，为负时是增加尺寸，
    # w1, h1宽度，高度方向结束的位置，
    img = img.crop((0, -blank, w_s, w_s - blank))
    return img