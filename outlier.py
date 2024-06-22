# import cv2
# import numpy as np

# # YOLOv5 model
# net = cv2.dnn.readNet("yolov5s.onnx", "")

# # Define the object class
# classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#            "hair drier", "toothbrush"]

# # Define the object class index
# class_index = 2  # car

# # Define the initial position, velocity, and acceleration
# x, y = 0, 0
# vx, vy = 0, 0
# ax, ay = 0, 0

# # Define the time step
# dt = 1 / 30  # 30 FPS

# # Define the video capture
# cap = cv2.VideoCapture("human.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect the object using YOLOv5
#     height, width, _ = frame.shape
#     blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 640), swapRB=True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(net.getUnconnectedOutLayersNames())

#     # Extract the bounding box coordinates
#     boxes = []
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if class_id == class_index and confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x1 = center_x - w // 2
#                 y1 = center_y - h // 2
#                 boxes.append([x1, y1, w, h])

#     # Update the position using the motion kinematics equation
#     if boxes:
#         x1, y1, w, h = boxes[0]
#         x = x1 + w // 2
#         y = y1 + h // 2
#         vx = (x - x) / dt
#         vy = (y - y) / dt
#         ax = (vx - vx) / dt
#         ay = (vy - vy) / dt
#     else:
#         x = x + vx * dt + 0.5 * ax * dt ** 2
#         y = y + vy * dt + 0.5 * ay * dt ** 2

#     # Draw the predicted position
#     cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

#     # Display the frame
#     cv2.imshow("Frame", frame)

#     # Exit on key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture
# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class PhysicsTracker:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.dt = 1/30  # Assuming 30 FPS

    def update(self, x, y):
        # Calculate acceleration (assuming constant acceleration)
        ax = (x - self.x - self.vx * self.dt) / (self.dt ** 2)
        ay = (y - self.y - self.vy * self.dt) / (self.dt ** 2)

        # Update velocity
        self.vx += ax * self.dt
        self.vy += ay * self.dt

        # Update position
        self.x = x
        self.y = y

    def predict(self):
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
    for result in results.xyxy:
        for detection in result:
            x, y, w, h, confidence, class_id = detection.tolist()
            if confidence > 0.5:  # Filter by confidence threshold
                x_center = int(x + w / 2)
                y_center = int(y + h / 2)

                # Update tracker
                tracker.update(x_center, y_center)

                # Predict next position
                x_pred, y_pred = tracker.predict()

                # Draw detection and prediction
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()