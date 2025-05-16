# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Initialize the YOLOv5 or YOLOv8 model
# net = cv2.dnn.readNet("yolov5s.onnx", "yolov5s.xml")

# # Define the classes for object detection
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Define the layer names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# # Load the video
# cap = cv2.VideoCapture("input_video.mp4")

# # Create a figure for 3D plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Lists to store the centroid coordinates
# xs, ys, zs = [], [], []

# def detect_and_track(frame):
#     # Get the frame dimensions
#     height, width, channels = frame.shape

#     # Create a blob from the frame
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#     # Set the blob as input to the network
#     net.setInput(blob)

#     # Get the detections
#     outs = net.forward(output_layers)

#     # Initialize the centroid coordinates
#     x, y, z = 0, 0, 0

#     # Loop through the detections
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Filter the detections by confidence and class
#             if confidence > 0.5 and classes[class_id] == "ball":
#                 # Get the bounding box coordinates
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Calculate the centroid coordinates
#                 x = center_x
#                 y = center_y
#                 z = 0  # Assuming 2D video, z-coordinate is 0

#                 # Draw the bounding box and centroid
#                 cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
#                 cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)

#     # Append the centroid coordinates to the lists
#     xs.append(x)
#     ys.append(y)
#     zs.append(z)

#     # Update the 3D plot
#     ax.clear()
#     ax.set_xlim(0, width)
#     ax.set_ylim(0, height)
#     ax.set_zlim(0, 10)
#     ax.plot(xs, ys, zs)

#     # Return the updated frame
#     return frame

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Detect and track the object
#     frame = detect_and_track(frame)

#     # Display the frame
#     cv2.imshow("Frame", frame)

#     # Update the 3D plot
#     plt.pause(0.001)

#     # Exit on key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the windows
# cap.release()
# cv2.destroyAllWindows()
# plt.show()



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import torch

# # Load YOLOv5 or YOLOv8 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Load video
# cap = cv2.VideoCapture('output.mp4')

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Lists to store centroid coordinates
# x_coords = []
# y_coords = []
# z_coords = []  # Since we're working with 2D images, z-coordinate will be 0

# plt.ion()  # Turn on interactive mode
# plt.show()

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     # Detect objects using YOLOv5 or YOLOv8
#     results = model(frame)
    
#     # Get the bounding box coordinates of the detected object
#     for i, pred in enumerate(results.xyxy[0]):
#         x1, y1, x2, y2, conf, cls = pred
        
#         # Calculate the centroid of the bounding box
#         centroid_x = (x1 + x2) / 2
#         centroid_y = (y1 + y2) / 2
        
#         # Append the centroid coordinates to the lists
#         x_coords.append(centroid_x)
#         y_coords.append(centroid_y)
#         z_coords.append(0)
        
#         # Draw the bounding box on the frame
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
#         # Draw the centroid on the frame
#         cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
        
#         # Display the frame with bounding box and centroid
#         cv2.imshow('Frame', frame)
        
#         # Clear the previous plot
#         ax.clear()
        
#         # Set the plot limits
#         ax.set_xlim(0, frame.shape[1])
#         ax.set_ylim(0, frame.shape[0])
#         ax.set_zlim(0, 1)
        
#         # Plot the centroid trajectory
#         ax.scatter(x_coords, y_coords, z_coords, c='r')
        
#         # Pause for a short time to update the plot
#         plt.pause(0.001)
        
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture('output.mp4')
centroids = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
  
    results = model(frame)
    if len(results.xyxy[0]) > 0:
        x1, y1, x2, y2, conf, cls = results.xyxy[0][0]
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        centroids.append([centroid_x, centroid_y])
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
        cv2.imshow('Frame', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the centroid coordinates to a CSV file
df = pd.DataFrame(centroids, columns=['x', 'y'])
df.to_csv('centroids.csv', index=False)

# Plot the centroid trajectory
plt.figure()
plt.plot([x for x, y in centroids], [y for x, y in centroids], 'r-')
plt.xlim(0, frame.shape[1])
plt.ylim(0, frame.shape[0])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()