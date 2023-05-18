from ultralytics import YOLO
import cv2
from torchvision import transforms
import torch
import numpy as np
from model import model
from torch import nn
import cvzone
import math

image_size = 256
classes = ['Alternariosis',
           'Anthracnose',
           'Bacteriosis',
           'Healthy',
            'Powdery Mildew',
           'Downey Mildew']

default_data_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Resize(size=(image_size, image_size)),

        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

img_path = 'h7.jpg' #place here your image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
model_yolo = YOLO('diplom/yolo') #place here your yola_model
#in case it crushes, rename file yolo to yolo.pt
results = model_yolo(img, show=False)
model_leaves = model
model_leaves.load_state_dict(torch.load('diplom/best_model', map_location=torch.device('cpu'))) # place here your classifier model
model_leaves.eval()

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop_img = img[y1:y2, x1:x2]
        crop_img = default_data_transform(crop_img)
        crop_img = crop_img.unsqueeze(0)
        y_pred = model_leaves(crop_img)
        arr = nn.Softmax(dim=1)(y_pred).cpu().detach().numpy()
        class_num = np.argmax(arr[0])
        class_conf = math.ceil(arr[0][class_num] * 100) / 100
        class_name = classes[class_num]
        crop_img = crop_img.squeeze()
        crop_img = crop_img.numpy()
        crop_img = crop_img / 2 + 0.5
        crop_img = np.transpose(crop_img, (1, 2, 0))
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=10, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{class_name} {class_conf}', (max(x1, 0), max(35, y1)),
                           scale=3 , thickness=2, offset=10)
cv2.imshow("Image", img)
cv2.waitKey(0)
