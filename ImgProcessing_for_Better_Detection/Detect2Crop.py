import cv2
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

# Load YOLOv5 model
model = attempt_load('Torch2PC\yolov5s.pt', map_location=torch.device('cpu'))

# Set confidence threshold and IOU threshold for object detection
conf_threshold = 0.5
iou_threshold = 0.5

# Load input image
image = cv2.imread('left005200.png')

# Convert image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image to fit YOLOv5 input size
input_size = model.hyperparams['input_size'][0]
image = cv2.resize(image, (input_size, input_size))

# Convert image to tensor
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

# Run object detection on input image
with torch.no_grad():
    detections = model(image)

# Apply non-max suppression to remove overlapping boxes
detections = non_max_suppression(detections, conf_threshold, iou_threshold)

# Iterate over detections and crop images
for detection in detections:
    if detection is not None:
        for x1, y1, x2, y2, conf, cls in detection:
            # Get center coordinates of detected object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get crop coordinates
            crop_size = 640
            crop_x1 = int(center_x - crop_size / 2)
            crop_y1 = int(center_y - crop_size / 2)
            crop_x2 = int(center_x + crop_size / 2)
            crop_y2 = int(center_y + crop_size / 2)

            # Crop image
            crop = image[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

            # Convert crop to numpy array and save image
            crop = crop.squeeze(0).permute(1, 2, 0).numpy()
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'crop_{cls}.jpg', crop)
