import argparse
import cv2
import time
import os
import json

from model import efficientnet_b0 as create_model
import torch
from torchvision import transforms

from PIL import Image

# from face_detection import RetinaFace
from GetEyes import detect


# from train import  get_non_ignored_params,get_fc_params,load_filtered_state_dict

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='D:/gaze estimation/L2CS-Net/models/L2CSNet_gaze360.pkl', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         # transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # detector = RetinaFace(gpu_id=0)
    x = 0

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            start_fps = time.time()

            # faces = detector(frame)
            faces = detect(frame)
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .70:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = data_transform(im_pil)
                    # img = Variable(img).cuda(gpu)
                    img = img.unsqueeze(0)
                    # read class_indict
                    json_path = './class_indices.json'
                    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

                    with open(json_path, "r") as f:
                        class_indict = json.load(f)

                    # create model
                    model = create_model(num_classes=10).to(device)
                    # load model weights
                    model_weight_path = "E:/yang/VIT/Test9_efficientNet/model-26.pth"
                    model.load_state_dict(torch.load(model_weight_path, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        # predict class
                        output = torch.squeeze(model(img.to(device))).cpu()
                        predict = torch.softmax(output, dim=0)
                        predict_cla = torch.argmax(predict).numpy()

                    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                                 predict[predict_cla].numpy())

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)

            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            success, frame = cap.read()
