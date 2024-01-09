import os
import sys
import json
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def read_split_data(root: str,val_root:str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    assert os.path.exists(root), "val dataset root: {} does not exist.".format(root)
    item_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # print('item_class_orginal:',item_class)
    # item_class_orginal: ['close', 'down', 'forward', 'left', 'leftdown', 'lefttop', 'right', 'rightdown', 'righttop',
    #                      'top']
    item_class.sort()
    # print(item_class)
    class_indices = dict((k, v) for v, k in enumerate(item_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    # print(class_indices)
    # class_indices:{'close': 0, 'down': 1, 'forward': 2, 'left': 3, 'leftdown': 4, 'lefttop': 5, 'right': 6, 'rightdown': 7, 'righttop': 8, 'top': 9}
    train_images_path = []
    train_images_label = []
    test_images_path = []
    test_images_label = []

    for cla in item_class:
        cla_path = os.path.join(root, cla)
        for i in os.listdir(cla_path):
            # print('clas_path:', root, cla)
            train_images_path.append(os.path.join(cla_path, i))
            image_class = class_indices[cla]
            train_images_label.append(image_class)

        val_path = os.path.join(val_root,cla)
        for j in os.listdir(val_path):
            test_images_path.append(os.path.join(val_path,j))
            # print('test:',test_images_path)
            val_lavel = class_indices[cla]
            test_images_label.append(val_lavel)
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for test.".format(len(test_images_path)))
    return train_images_path, train_images_label, test_images_path, test_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def  train_one_epoch(model, optimizer, data_loader, device, epoch,df_train_log,batch_idx=0):
    model.train()
    log_train1 = {}
    meanloss = []
    predsss = []
    lab = []
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        # print('labels:',labels)
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # 获取当前 batch 的标签类别和预测类别
        _, preds = torch.max(pred, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        meanloss.append(loss)
        predsss.extend(preds)
        lab.extend(labels)
        ## 新增修改
        log_train = {}
        log_train['epoch'] = epoch
        log_train['batch'] = batch_idx
        # 计算分类评估指标
        log_train['train_loss'] = loss
        log_train['train_accuracy'] = accuracy_score(labels, preds)
        # log_train['train_precision'] = precision_score(labels, preds, average='macro')
        # log_train['train_recall'] = recall_score(labels, preds, average='macro')
        # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')
        df_train_log = df_train_log.append(log_train, ignore_index=True)
    log_train1['batch'] = epoch
    log_train1['train_loss'] = np.mean(meanloss)
    log_train1['train_accuracy'] = accuracy_score(lab, predsss)
    return mean_loss.item(),df_train_log,log_train1


@torch.no_grad()
def evaluate(model, data_loader, device,epoch=0):
    model.eval()

    model.eval()
    loss_list = []
    labels_list = []
    preds_list = []

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        outputs = model(images.to(device))
        pred = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

        # 获取整个测试集的标签类别和预测类别
        preds = pred  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.to(device))  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        loss_list.append(loss)
        labels_list.extend(labels)
        preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

    return sum_num.item() / total_num,log_test
