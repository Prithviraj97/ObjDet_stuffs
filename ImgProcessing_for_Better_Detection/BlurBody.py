import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Define the blur kernel size and strength
kernel_size = (23, 23)
blur_strength = 30

cap = cv2.VideoCapture('video.avi')
while True:
    ret, frame = cap.read()
    results = model(frame)

    person_objects = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    for person in person_objects:
        # Extract the bounding box coordinates of the detected person
        x1, y1, x2, y2, conf, cls = person.int().tolist()
        body = frame[y1:y2, x1:x2]
        blurred_body = cv2.GaussianBlur(body, kernel_size, blur_strength)

        # Replace the body region with the blurred version
        frame[y1:y2, x1:x2] = blurred_body
    cv2.imshow('frame', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
