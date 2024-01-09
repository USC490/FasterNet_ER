from __future__ import print_function,division

import collections
import copy
import os
import math
import argparse
import timm.models
import torchvision.models as models
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
from resnetmodel import resnet50
from torchvision import models
# from a_learn.t import FasterNet
from MyNet import FasterNet
# 忽略红色提示
import warnings
warnings.filterwarnings("ignore")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(args)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path,args.val_path)
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([
                                     transforms.Resize((num_model,num_model)),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                     # transforms.RandomRotation(degrees=30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((num_model,num_model)),
                                   # transforms.RandomRotation(degrees=30),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    # model = create_model(num_classes=args.num_classes).to(device)
    # model = timm.models.vgg16(pretrained=False,num_classes=10).to(device)
    # model = models.shufflenet_v2_x1_0(pretrained=False,num_classes=10).to(device)
    # model = models.resnet34(pretrained=False,num_classes=10).to(device)
    model = FasterNet(args.num_classes).to(device)
    print(model)
    weights_dict = torch.load(args.weight)
    del_key = []
    add_key = []
    for key, va in weights_dict.items():
        # print('shape:',key,va.shape)
        if 'head' in key:
            del_key.append(key)
        if 'patch_embed' or 'stages' or 'avgpool_pre_head' in key:
            add_key.append(key)
    # 删除字典中的classifier层
    # items()方法用于返回字典中所有的键值对
    for key in del_key:
        if key == 'avgpool_pre_head.1.weight':
            pass
        else:
            del weights_dict[key]
    weights_dict1 = copy.deepcopy(weights_dict)
    for key in add_key:
        # print(key)
        if 'patch_embed' in key:
            new_key = key.replace('patch_embed', 'patch_embed1')
            weights_dict1[new_key] = weights_dict[key]
            del weights_dict1[key]
        if 'stages' in key:
            new_key = key.replace('stages', 'stages1')
            weights_dict1[new_key] = weights_dict[key]
            del weights_dict1[key]
        if 'avgpool_pre_head' in key:
            new_key = key.replace('avgpool_pre_head', 'avgpool_pre_head1')
            weights_dict1[new_key] = weights_dict[key]
            del weights_dict1[key]
    # 字典合并
    weights_dict = dict(weights_dict, **weights_dict1)
    ##strict=False不写的话，默认为true
    weights_dict = collections.OrderedDict(weights_dict)
    model.load_state_dict(weights_dict, strict=False)


    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    # 查看是否某层是否冻结
    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    batch_idx = 0
    best_test_accuracy = 0
    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    log_train = {}
    df_train_log1 = pd.DataFrame()
    log_train['epoch'] = 0
    log_train['batch'] = 0

    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['epoch'] = 0
    for epoch in range(args.epochs):
        # train
        batch_idx +=1
        mean_loss,log,log1 = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch+1,
                                    batch_idx=batch_idx,
                                    df_train_log= df_train_log)
        # df_train_log = df_train_log.append(log_train, ignore_index=True)
        df_train_log = log
        df_train_log1 = df_train_log1.append(log1,ignore_index=True)
        scheduler.step()

        # validate
        acc,log_test = evaluate(model=model,
                       data_loader=val_loader,
                       device=device,
                       epoch=epoch+1)
        df_test_log = df_test_log.append(log_test,ignore_index=True)
        print("[epoch {}] accuracy: {}".format(epoch+1, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]


        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        if log_test['test_accuracy'] > best_test_accuracy:
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = 'weights/eyechi-{:.3f}.pth'.format(best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path = 'weights/eyechi-{:.3f}.pth'.format(log_test['test_accuracy'])
            # torch.save(model, new_best_checkpoint_path)
            torch.save(model.state_dict(), new_best_checkpoint_path)
            print('保存新的最佳模型', 'weights/eyechi-{:.3f}.pth'.format(best_test_accuracy))
            # best_test_accuracy = log_test['test_accuracy']

        df_train_log.to_csv('训练日志-训练集eyechi.csv', index=False)
        df_train_log1.to_csv('训练日志-训练集eyechi.csv', index=False)
        df_test_log.to_csv('训练日志-测试集eyechi.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="E:\\eye\\train")
    parser.add_argument('--val-path', type=str,
                        default="E:\\eye\\val")

    parser.add_argument('--weights', type=str, default='E:\\my_Fasternet\\MyFaster\\weights\\fca(9)-0.950.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()


    main(opt)
