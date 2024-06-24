import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# Set up the Streamlit app
st.title("Object Detection and Tracking")

# Upload the video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

# Upload the YOLOv5 weights
weights_file = st.file_uploader("Upload YOLOv5 weights", type=["pt"])

# Function to load the YOLOv5 model
def load_model(weights_file):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file)
    return model

# Function to perform object detection
def detect_objects(model, frame):
    results = model(frame)
    return results

# Function to track objects
def track_objects(results, frame):
    # Get the bounding box coordinates and class IDs
    boxes = results.xyxy[0].numpy()
    class_ids = results.xyxy[0].numpy()[:, -1].astype(int)

    # Create a list to store the tracked objects
    tracked_objects = []

    # Iterate over the bounding boxes
    for box, class_id in zip(boxes, class_ids):
        x, y, x2, y2 = box[:4].astype(int)
        center_x, center_y = (x + x2) // 2, (y + y2) // 2

        # Append the tracked object to the list
        tracked_objects.append((center_x, center_y))

    return tracked_objects

# Function to construct the trajectory
def construct_trajectory(tracked_objects):
    trajectory = np.array(tracked_objects)
    return trajectory

# Function to plot the trajectory in 3D
def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], np.arange(len(trajectory)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Frame')
    return fig

# Main function
def main():
    if video_file is not None and weights_file is not None:
        # Load the video
        video = cv2.VideoCapture(video_file.name)

        # Load the YOLOv5 model
        model = load_model(weights_file.name)

        # Create a list to store the tracked objects
        tracked_objects = []

        # Iterate over the video frames
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Perform object detection
            results = detect_objects(model, frame)

            # Track objects
            objects = track_objects(results, frame)
            tracked_objects.extend(objects)

            # Display the output
            st.image(cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB))

        # Construct the trajectory
        trajectory = construct_trajectory(tracked_objects)

        # Plot the trajectory in 3D
        fig = plot_trajectory(trajectory)
        st.pyplot(fig)

if __name__ == "__main__":
    main()