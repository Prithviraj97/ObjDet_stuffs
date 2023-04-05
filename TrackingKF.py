import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter

# Initialize the Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0, 0, 0, 0]) # initial state
kf.P *= 10 # initial uncertainty
kf.R = np.array([[1, 0], [0, 1]]) # measurement noise

# Initialize the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)#torch.hub.load('ultralytics/yolov5', 'custom', path='cus.pt', force_reload=True)
# torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()
    confidences = results.xyxy[0][:, 4].cpu().numpy()
    class_ids = results.xyxy[0][:, 5].cpu().numpy()
    boxes = boxes[confidences > 0.3]
    
    # Update the Kalman filter with the detected object's position
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        z = np.array([[x + w/2], [y + h/2]])
        kf.predict()
        kf.update(z)
        state = kf.x
        
    # Draw the bounding box on the frame
    x, y, w, h = state.astype('int')
    cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
