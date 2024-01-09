import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general1 import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def warp_affine(points,Image,s=1.0):
    flag = 0
    new_points = np.zeros((5,2))
    #   np.sqrt(np.square(points[1][i] - points[0][i]) + np.square(points[6][i] - points[5][i]
    eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
    dy = points[1][1] - points[0][1]
    dx = points[1][0] - points[0][0]
    # print('dy:',dy,
    #       'dx:',dx)
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx)
    print('angle:',angle)
    flag = 1
    rot = cv2.getRotationMatrix2D(eye_center, angle, scale=s)
    # print('rot:',rot)
    rot_img = cv2.warpAffine(Image, rot, dsize=(Image.shape[1], Image.shape[0]))
    for i in range(5):
        new_points[i][0] = rot[0][0] * points[i][0] + rot[0][1] * points[i][1] + rot[0][2]
        new_points[i][1] = rot[1][0] * points[i][0] + rot[1][1] * points[i][1] + rot[1][2]
    return rot_img, new_points, flag

def detect(opt):
    num = 0
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print(pred[...,4].max())

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()
        # print('pred:',pred[0].shape,pred) pred[0]是张量，torch.Size([人脸数量, 21维向量])

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, model, img, im0s)

        # Process detections
        # p是图片的路径， im0是图片,只要检测到人脸，列表pred的长度始终为1
        # 图片裁剪使用img[y0:y1, x0:x1]，其中参数分别代表y轴和x轴的起始坐标
        for i, det in enumerate(pred):  # detections per image
            # p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p, s, im0= path, '', im0s.copy()

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # save_leftEye = str(save_dir /'lefteye')
            # save_rightEye = str(save_dir /'righteye')
            # save_twoEye = str(save_dir/ 'two_eyes')

            save_leftEye1 = str(save_dir /'Lefteye')
            save_rightEye1 = str(save_dir /'Righteye')
            save_twoEye1 = str(save_dir /'Two_eyes')
            # if not os.path.exists(save_leftEye):
            #     os.mkdir(save_leftEye)
            # if not os.path.exists(save_rightEye):
            #     os.mkdir(save_rightEye)
            # if not os.path.exists(save_twoEye):
            #     os.mkdir(save_twoEye)
            if not os.path.exists(save_leftEye1):
                os.mkdir(save_leftEye1)
            if not os.path.exists(save_rightEye1):
                os.mkdir(save_rightEye1)
            if not os.path.exists(save_twoEye1):
                os.mkdir(save_twoEye1)

            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det)==0:
                print('图片'+p.name+'未检测到人脸')

            if len(det):
                print('图片%s检测到%d张人脸' % (p.name, det.shape[0]))
                # Rescale boxes from img_size to im0 size，  img.shape为（1，3，640，640）640是传入维度
                # img.shape[2:]是（640，640） det[:,:4]是（人脸数量，4） , im0.shape是（h,w,3）
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Write results, det[:,:6].shape是（人脸数量，6）reversed这里是翻转的0维
                # print('det:',det[:,:6],reversed(det[:,:6]))
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    # print(det[:,:6])
                    # print('(*xyxy, conf, cls):',*xyxy)
                    # print('conf:', conf)
                    # print('cls:', cls)
                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        kpts = det[det_index, 6:]
                        # im0.shape[:2]是（h,w)
                        Image = im0.copy()
                        points = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        ## 是否旋转图片
                        new_img,new_points,new_flage=warp_affine(points,Image, s = 1.0)
                        if new_flage == 1:
                            points = new_points
                            Image = new_img
                        ## 左右眼距离
                        # cv2.imwrite(os.path.join('C:/Users/Admin/Desktop/2',p.name), new_img)

                        # cv2.imshow('test',new_img)
                        # cv2.waitKey(0)
                        leftToRightDist = int(
                            np.sqrt(np.square(points[1][0] - points[0][0]) + np.square(points[1][1] - points[0][1])))
                        img_size = np.asarray(im0.shape)[0:2]
                        imageW = img_size[1]
                        imageH = img_size[0]
                        # 左眼框的左上角坐标
                        leftEyeLX = max(0, int(points[0][0] - leftToRightDist / 3))
                        leftEyeLY = max(0, int(points[0][1] - leftToRightDist / 6))
                        # cv2.circle(im0, (leftEyeLX, leftEyeLY), 4, (0, 0, 0), -1)

                        # 眼睛正方形框截取
                        leftEyeLY1 = max(0, int(points[0][1] - leftToRightDist / 3))
                        # print('左眼上坐标：', leftEyeLX, leftEyeLY)
                        # 左眼框的右下角坐标
                        leftEyeRX = min(imageW, int(points[0][0] + leftToRightDist / 3))
                        leftEyeRY = min(imageH, int(points[0][1] + leftToRightDist / 6))
                        # cv2.circle(im0, (leftEyeRX, leftEyeRY), 4, (0, 0, 0), -1)

                        leftEyeRY1 = min(imageH, int(points[0][1] + leftToRightDist / 3))
                        # print('左眼下坐标：', leftEyeRX, leftEyeRY)

                        # 右眼框的左上角坐标
                        rightEyeLX = max(0, int(points[1][0] - leftToRightDist / 3))
                        rightEyeLY = max(0, int(points[1][1] - leftToRightDist / 6))
                        # cv2.circle(im0, (rightEyeLX, rightEyeLY), 4, (0, 0, 0), -1)

                        rightEyeLY1 = max(0, int(points[1][1] - leftToRightDist / 3))
                        # print('右眼上坐标：', rightEyeLX, rightEyeLY)
                        # 右眼框的右下角坐标
                        rightEyeRX = min(imageW, int(points[1][0] + leftToRightDist / 3))
                        rightEyeRY = min(imageH, int(points[1][1] + leftToRightDist / 6))
                        # cv2.circle(im0, (rightEyeRX, rightEyeRY), 4, (0, 0, 0), -1)

                        rightEyeRY1 = min(imageH, int(points[1][1] + leftToRightDist / 3))
                        # print('右眼下坐标：', rightEyeRX, rightEyeRY)
                        # cv2.imshow('123',im0)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # 抠出左眼框图像数据
                        leftEyeImage = Image[leftEyeLY:leftEyeRY, leftEyeLX:leftEyeRX, :]
                        leftEyeImage = Image[leftEyeLY:leftEyeRY, leftEyeLX:leftEyeRX, :]

                        leftEyeImage1 = Image[leftEyeLY1:leftEyeRY1, leftEyeLX:leftEyeRX, :]
                        # cv2.imwrite(os.path.join(leftEyePath, leftEyePath + str(index) + '.jpg'), leftEyeImage1)
                        # 抠出右眼框图像数据
                        # lrightEyeImage = images[rightEyeLY:rightEyeRY, rightEyeLX:rightEyeRX, :]
                        lrightEyeImage = Image[rightEyeLY:rightEyeRY, rightEyeLX:rightEyeRX, :]
                        lrightEyeImage1 = Image[rightEyeLY1:rightEyeRY1, rightEyeLX:rightEyeRX, :]

                        # x, y = p.name.split('.')
                        # print(str(det.shape[0]),str(det_index),x)
                        eyesImage = Image[leftEyeLY:rightEyeRY, leftEyeLX:rightEyeRX, :]
                        eyesImage1 = Image[leftEyeLY:rightEyeRY1, leftEyeLX:rightEyeRX, :]

                        # print('图片形状：',leftEyeImage.shape,lrightEyeImage.shape,eyesImage.shape)

                        # cv2.imwrite(os.path.join(save_leftEye, 'leftEye' + str(det.shape[0])+'_'+ str(det_index)  +'_'+ x+ '.jpg'), leftEyeImage)
                        # cv2.imwrite(os.path.join(save_rightEye, 'rightEye' + str(det.shape[0])+'_'+ str(det_index)  + '_' +x+ '.jpg'), lrightEyeImage)
                        # cv2.imwrite(os.path.join(save_twoEye, 'eyes' + str(det.shape[0])+'_'+ str(det_index) + '_' +x+ '.jpg', ), eyesImage)
                        #
                        # cv2.imwrite(os.path.join(save_leftEye1, 'leftEye' + str(det.shape[0])+'_'+str(det_index)  + '_' + x+ '(1).jpg'), leftEyeImage1)
                        # cv2.imwrite(os.path.join(save_rightEye1, 'rightEye' + str(det.shape[0])+'_'+ str(det_index)   + '_'+x+ '(1).jpg'), lrightEyeImage1)
                        # cv2.imwrite(os.path.join(save_twoEye1, 'eyes' + str(det.shape[0])+'_'+ str(det_index)  + '_' + x+ '(1).jpg', ), eyesImage1)
                        # print('图片名称：',x)
                        ren = '1'
                        # cv2.imwrite(os.path.join(save_leftEye1,  str(num) + '.jpg'), leftEyeImage1)
                        # cv2.imwrite(os.path.join(save_rightEye1, str(num) + '.jpg'), lrightEyeImage1)
                        # cv2.imwrite(os.path.join(save_twoEye1,  str(num) + '.jpg', ), eyesImage1)

                        # cv2.imwrite(os.path.join(save_leftEye1, 'LP'+ren + str(num) + '.jpg'), leftEyeImage1)
                        # cv2.imwrite(os.path.join(save_rightEye1, 'RP'+ren + str(num) + '.jpg'), lrightEyeImage1)
                        # cv2.imwrite(os.path.join(save_twoEye1, 'P'+ren + str(num) + '.jpg', ), eyesImage1)
                        cv2.imwrite(os.path.join('c:', 'dl' + str(num) + '.jpg', ), eyesImage1) # 保存双眼图像位置
                        # with open('label.txt', 'a') as file:
                        #     file.write(x+'.png'+' '+'top_look/Lefteye/LP'+ren + str(num) + '.jpg'+' '+'top_look/Righteye/RP'+ren + str(num) + '.jpg'+' '+'top_look/Two_eyes/P'+ren + str(num) + '.jpg'+' 9'+'\n')
                        # 命名格式 ：eye+图片检测到人脸数量_第几张_图片名
                        num += 1
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model.pt path(s)') #yolov7-w6-face.pt权重路径  权重下载https://github.com/clibdev/yolov7-face
    parser.add_argument('--source', type=str, default='C:\\Users', help='source')  # 要处理的图片路径
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')#0.5
    parser.add_argument('--iou-thres', type=float, default=0.8, help='IOU threshold for NMS')#0.45
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', type=int, default=5, help='number of keypoints')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
