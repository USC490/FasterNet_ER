import os
# 重命名文件

# 图片所在路径
root_path = " "
filename_list = os.listdir(root_path)
num = 0
for filename in filename_list:
    if filename.endswith('.jpg'):
        src_img_path = os.path.join(os.path.abspath(root_path), filename)
        new_img_code = filename.split('.')

        print(new_img_code)
        dst_img_path = os.path.join(os.path.abspath(root_path),  str(num)+'.jpg')
        print(dst_img_path)
        try:
            os.rename(src_img_path, dst_img_path)
            print('converting %s to %s ...' % (src_img_path, dst_img_path))
        except:
            continue
        num += 1
print(num)