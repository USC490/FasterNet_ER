import json
import os
from tqdm import tqdm
from torchvision import models
import numpy as np
import pandas as pd
from build_model.resnet_model import resnet18
from PIL import Image
from a_learn.Fast import FasterNet
import torch
import torch.nn.functional as F
from torchvision import datasets
from model import efficientnet_b0 as create_model
import torchvision.models as models
# 忽略红色提示
import warnings
warnings.filterwarnings("ignore")
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 图像预处理
from torchvision import transforms
# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
## 数据集文件夹路径
dataset_dir = 'E:'
test_path = os.path.join(dataset_dir, 'val')
from torchvision import datasets
# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)
# 载入类别名称 和 ID索引号 的映射字典
idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
# print(np.load('labels_to_idx.npy', allow_pickle=True).item())
## 载入类别名称 和 ID索引号 的映射字典
## 表格A-测试集图像路径及标注
img_paths = [each[0] for each in test_dataset.imgs]
df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = test_dataset.targets
df['标注类别名称'] = [idx_to_labels[ID] for ID in test_dataset.targets]


##导入训练好的模型
model = FasterNet(num_classes=10).to(device)

# 加载权重文件
model.load_state_dict(torch.load('weights\\', map_location=device))

# print(model)
model.eval()

## 表格B-测试集每张图像的图像分类预测结果，以及各类别置信度
# 记录 top-n 预测结果
n = 3
df_pred = pd.DataFrame()
for idx, row in tqdm(df.iterrows()):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    w, h = img_pil.size

    # w, h = img_pil.size
    blank = (w - h) / 2
    img_pil = img_pil.crop((0, -blank, w, w - blank))
    input_img = test_transform(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_img)

    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    pred_dict = {}
    # print(row)
    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
    # print('pred_ids:',pred_ids)
    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict['top-{}-预测ID'.format(i)] = pred_ids[i - 1]
        pred_dict['top-{}-预测名称'.format(i)] = idx_to_labels[pred_ids[i - 1]]
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()

    df_pred = df_pred.append(pred_dict, ignore_index=True)
pd.set_option('display.max_columns', 1000) #最大行数
pd.set_option('display.width', 1000) #字段宽度
pd.set_option('display.max_colwidth', 1000) #字段内容宽度
##拼接AB两张表格
df = pd.concat([df, df_pred], axis=1)
print(df_pred)
df.to_csv('测试集预测结果.csv', index=False)