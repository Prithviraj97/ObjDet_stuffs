'''The code will use YOLOv5 to detect the person in the frame and crop the image across height and width from the center of the 
detected person.'''
import cv2
import pandas as pd
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.80
model.iou = 0.5
model.classes=[0]
image = cv2.imread('left004441.png')
results = model(image)
boxes = results.xyxy[0]

x1, y1, x2, y2,conf, cls = results.xyxy[0][0].int().tolist()
# body = image[y1:y2, x1:x2]
center_x = int((x1 + x2) / 2)
center_y = int((y1 + y2) / 2)

crop_size = 320
crop_x1 = int(center_x - crop_size / 2)
crop_y1 = int(center_y - crop_size / 2)
crop_x2 = int(center_x + crop_size / 2)
crop_y2 = int(center_y + crop_size / 2)
body = image[crop_y1:crop_y2, crop_x1:crop_x2]
print("The crop dimensions: ", crop_x1, crop_y1, crop_x2, crop_y2)
print("                        xmin ymin xmax ymax")
print("The bounding boxes are: ", x1, y1, x2, y2)
# cv2.imshow("Frame", body)
body2 = cv2.resize(body, (640, 640))
cv2.imwrite("cropleft004441resized3.png", body2)
cv2.waitKey(0)
cv2.destroyAllWindows()