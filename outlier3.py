import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class PhysicsTracker:
    def __init__(self):
        self.x = None
        self.y = None
        self.vx = None
        self.vy = None
        self.dt = 1/30  # Assuming 30 FPS

    def initialize(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0

    def update(self, x, y):
        if self.x is None or self.y is None:
            self.initialize(x, y)
        else:
            # Calculate velocity
            self.vx = (x - self.x) / self.dt
            self.vy = (y - self.y) / self.dt

            # Update position
            self.x = x
            self.y = y

    def predict(self):
        if self.x is None or self.y is None:
            return None, None
        else:
            # Predict next position using kinematics equation
            x_pred = self.x + self.vx * self.dt
            y_pred = self.y + self.vy * self.dt

            return x_pred, y_pred

# Initialize tracker
tracker = PhysicsTracker()

# Initialize video capture
cap = cv2.VideoCapture('output.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv5
    results = model(frame)
    detections = results.xyxy[0].tolist()
    if detections:
        # Filter by confidence threshold and person class
        person_detections = [detection for detection in detections if detection[4] > 0.5 and detection[5] == 0]
        if person_detections:
            # Get the detection with the highest confidence
            person_detection = max(person_detections, key=lambda x: x[4])
            x, y, w, h = person_detection[:4]
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)

            # Update tracker
            tracker.update(x_center, y_center)

            # Draw detection
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

            # Predict next position
            x_pred, y_pred = tracker.predict()
            if x_pred is not None and y_pred is not None:
                cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()