import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import math
import cv2
import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='SimHei') # 中文字体
## 载入类别名称和ID

idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
# print(classes)

## 载入测试集预测结果表格
df = pd.read_csv('测试集预测结果.csv')
# print(df.head())

## 生成混淆矩阵
confusion_matrix_model = confusion_matrix(df['标注类别名称'], df['top-1-预测名称'])
# print(confusion_matrix_model.shape)

##可视化混淆矩阵
def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    """
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    """
    # plt.figure(figsize=(10, 10))
    plt.figure(figsize=(8, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(len(classes))

    plt.title('混淆矩阵', fontsize=30)
    plt.xlabel('预测类别', fontsize=25, c='r')
    plt.ylabel('真实类别', fontsize=25, c='r')
    plt.tick_params(labelsize=16)  # 设置类别文字大小
    plt.xticks(tick_marks, classes, rotation=90)  # 横轴文字旋转
    plt.yticks(tick_marks, classes)

    # 写数字
    # plt.text(x=2.2,  # 文本x轴坐标
    #          y=8,  # 文本y轴坐标
    #          s='basic unility of text',  # 文本内容
    #          rotation=1,  # 文字旋转
    #          ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
    #          va='baseline',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
    #          fontdict=dict(fontsize=12, color='r',
    #                        family='monospace',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
    #                        weight='bold',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
    #
    #                        )  # 字体属性设置
    #          )
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=12)

    plt.tight_layout()

    # plt.savefig('混淆矩阵.pdf', dpi=300)  # 保存图像
    plt.show()
## 显示混淆矩阵
## 精选配色方案 Blues BuGn Reds Greens Greys binary Oranges Purples BuPu GnBu OrRd RdPu
cnf_matrix_plotter(confusion_matrix_model, classes, cmap='Blues')

## 筛选出测试集中，真实为A类，但被误判为B类的图像
true_A = 'down'
pred_B = 'forward'

wrong_df = df[(df['标注类别名称']==true_A)&(df['top-1-预测名称']==pred_B)]
# print( df[(df['标注类别名称']==true_A)])
# print(wrong_df)

## 可视化上表中所有被误判的图像
for idx, row in wrong_df.iterrows():
    img_path = row['图像路径']
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    title_str = img_path + '\nTrue:' + row['标注类别名称'] + ' Pred:' + row['top-1-预测名称']
    plt.title(title_str)
    plt.show()