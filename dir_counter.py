from ultralytics import YOLO
import cv2
from torchvision import transforms
import torch
import numpy as np
from model import model
from torch import nn
import cvzone
import math
import os

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
total = 0

dir_path = '/Users/nikitakocherin/workspace/plant_diseases_4/train/Здоровые' #place here dir_path
dict_plants = {'Alternariosis': 0,
               'Anthracnose': 0,
               'Bacteriosis': 0,
               'Healthy': 0,
               'Powdery Mildew': 0,
               'Downey Mildew': 0}
model_yolo = YOLO('diplom/yolo') #place here your yola_model
#in case it crushes, rename yolo to yolo.pt
model_leaves = model
model_leaves.load_state_dict(torch.load('diplom/best_model', map_location=torch.device('cpu'))) # place here your classifier model
model_leaves.eval()

for filename in os.listdir(dir_path):
    if filename[-4:] != 'jpeg' and filename[-3:] != 'jpg':
        continue
    f = os.path.join(dir_path, filename)
    # checking if it is a file
    if os.path.isfile(f):
        img_path = f
        print(f)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        results = model_yolo(img, show=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                total += 1
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
                dict_plants[class_name] += 1
                crop_img = crop_img.squeeze()
                crop_img = crop_img.numpy()
                crop_img = crop_img / 2 + 0.5
                crop_img = np.transpose(crop_img, (1, 2, 0))
print('total: ', total, ', illness_dict: ', dict_plants, sep='')
