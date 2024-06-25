# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import torch

# # Load YOLOv5 or YOLOv8 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Load video
# cap = cv2.VideoCapture('human.mp4')

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Lists to store centroid coordinates
# x_coords = []
# y_coords = []
# z_coords = []  # Since we're working with 2D images, z-coordinate will be 0

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     # Detect objects using YOLOv5 or YOLOv8
#     results = model(frame)
    
#     # Get the bounding box coordinates of the detected object
#     for pred in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = pred
        
#         # Calculate the centroid of the bounding box
#         centroid_x = (x1 + x2) / 2
#         centroid_y = (y1 + y2) / 2
        
#         # Append the centroid coordinates to the lists
#         x_coords.append(centroid_x)
#         y_coords.append(centroid_y)
#         z_coords.append(0)
        
#         # Plot the centroid on the 3D plot
#         ax.scatter(centroid_x, centroid_y, 0, c='r')
        
#         # Draw the bounding box on the frame
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
#         # Display the frame with bounding box
#         cv2.imshow('Frame', frame)
        
#         # Update the 3D plot
#         plt.pause(0.001)
#         plt.clf()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(x_coords, y_coords, z_coords, c='r')
        
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import csv

# Load YOLOv5 or YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
cap = cv2.VideoCapture('output.mp4')

# Create a CSV file to store centroid coordinates
with open('centroid_coordinates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y"])

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect objects using YOLOv5 or YOLOv8
    results = model(frame)
    
    # Get the bounding box coordinates of the detected object
    if len(results.xyxy[0]) > 0:
        x1, y1, x2, y2, conf, cls = results.xyxy[0][0]
        
        # Calculate the centroid of the bounding box
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        
        # Save the centroid coordinates to the CSV file
        with open('centroid_coordinates.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([centroid_x.item(), centroid_y.item()])
        
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw the centroid on the frame
        cv2.circle(frame, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
        
        # Display the frame with bounding box and centroid
        cv2.imshow('Frame', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Load the centroid coordinates from the CSV file
x_coords = []
y_coords = []
with open('centroid_coordinates.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        x_coords.append(float(row[0]))
        y_coords.append(float(row[1]))

# Create an animation of the centroid trajectory
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, max(x_coords))
ax.set_ylim(0, max(y_coords))

def animate(i):
    ax.clear()
    ax.set_xlim(0, max(x_coords))
    ax.set_ylim(0, max(y_coords))
    ax.scatter(x_coords[:i+1], y_coords[:i+1], c='r')

ani = animation.FuncAnimation(fig, animate, frames=len(x_coords), interval=50)

plt.show()