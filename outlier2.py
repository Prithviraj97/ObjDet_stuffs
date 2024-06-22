# import cv2
# import torch
# import numpy as np

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# class PhysicsTracker:
#     def __init__(self):
#         self.x = None
#         self.y = None
#         self.vx = None
#         self.vy = None
#         self.dt = 1/30  # Assuming 30 FPS
#         self.initialized = False

#     def initialize(self, x, y):
#         self.x = x
#         self.y = y
#         self.vx = 0
#         self.vy = 0
#         self.initialized = True

#     def update(self, x, y):
#         if self.initialized:
#             # Calculate velocity
#             self.vx = (x - self.x) / self.dt
#             self.vy = (y - self.y) / self.dt

#             # Update position
#             self.x = x
#             self.y = y
#         else:
#             self.initialize(x, y)

#     def predict(self):
#         if self.initialized:
#             # Predict next position using kinematics equation
#             x_pred = self.x + self.vx * self.dt
#             y_pred = self.y + self.vy * self.dt

#             return x_pred, y_pred
#         else:
#             return None, None

# # Initialize tracker
# tracker = PhysicsTracker()

# # Initialize video capture
# cap = cv2.VideoCapture('output.mp4')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect objects using YOLOv5
#     results = model(frame)
#     for result in results.xyxy:
#         for detection in result:
#             x, y, w, h, confidence, class_id = detection.tolist()
#             if confidence > 0.5 and class_id == 0:  # Filter by confidence threshold and class (person)
#                 x_center = int(x + w / 2)
#                 y_center = int(y + h / 2)

#                 # Update tracker
#                 tracker.update(x_center, y_center)

#                 # Draw detection
#                 cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
#                 cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

#                 # Predict next position
#                 x_pred, y_pred = tracker.predict()
#                 if x_pred is not None and y_pred is not None:
#                     cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 0, 255), -1)

#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


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
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv5
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Filter by confidence threshold and person class
    person_detections = detections[(detections['confidence'] > 0.5) & (detections['class'] == 0)]

    if not person_detections.empty:
        # Get the first person detection (assuming there is only one person)
        person_detection = person_detections.iloc[0]

        x = int(person_detection['xmin'])
        y = int(person_detection['ymin'])
        w = int(person_detection['xmax'] - person_detection['xmin'])
        h = int(person_detection['ymax'] - person_detection['ymin'])

        x_center = x + w // 2
        y_center = y + h // 2

        # Update tracker
        tracker.update(x_center, y_center)

        # Draw detection
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
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