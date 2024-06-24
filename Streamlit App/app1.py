import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to perform object detection
def detect_objects(uploaded_video, weights):
    # Initialize the video capture
    video = cv2.VideoCapture(uploaded_video)

    # Initialize the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Lists to store the x, y, and z coordinates
    x_coords = []
    y_coords = []
    z_coords = []

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Iterate over the detected objects
        for obj in results.xyxy[0]:
            # Get the coordinates of the detected object
            x, y, x2, y2 = obj[:4]
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

            # Calculate the center coordinates
            center_x = (x + x2) // 2
            center_y = (y + y2) // 2

            # Append the coordinates to the lists
            x_coords.append(center_x)
            y_coords.append(center_y)
            z_coords.append(video.get(cv2.CAP_PROP_POS_FRAMES))

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        st.image(frame)

        # Plot the trajectory
        ax.scatter(x_coords, y_coords, z_coords)

    # Display the plot
    st.pyplot(fig)

# Streamlit app
st.title("Object Detection and Tracking")

# File uploader for the video
uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi'])

# File uploader for the YOLOv5 weights
weights = st.file_uploader("Choose YOLOv5 weights", type=['pt'])

if uploaded_video and weights:
    detect_objects(uploaded_video, weights)