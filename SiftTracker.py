# import cv2
# import numpy as np

# # Load the reference image and extract keypoints and descriptors
# ref_img = cv2.imread('left005200.png')
# gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# ref_kp, ref_des = sift.detectAndCompute(gray, None)

# # Set up the feature matcher and object tracker
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# # tracker = cv2.Tracker_create('KCF')

# # Set up the video capture
# cap = cv2.VideoCapture('video.avi')

# # Initialize the object bounding box
# bbox = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the current frame to grayscale and extract keypoints and descriptors
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     kp, des = sift.detectAndCompute(gray, None)

#     # Match the keypoints and descriptors with those from the reference image
#     matches = bf.match(ref_des, des)
#     matches = sorted(matches, key=lambda x: x.distance)
    
#     # Extract the matched keypoints
#     ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#     # Compute the homography matrix
#     H, mask = cv2.findHomography(ref_pts, pts, cv2.RANSAC, 5.0)

#     # Project the bounding box of the object from the reference image onto the current frame
#     if bbox is not None:
#         bbox = cv2.perspectiveTransform(np.array([bbox]), H)[0]

#     # Update the object tracker with the new bounding box
#     ok = tracker.update(frame, bbox)
#     if ok:
#         bbox = tracker.get_position()
#         bbox = (int(bbox.left()), int(bbox.top()), int(bbox.width()), int(bbox.height()))
#         cv2.rectangle(frame, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
#     else:
#         # If the tracker fails, try to re-initialize it with the new bounding box
#         bbox = cv2.selectROI(frame, False)
#         tracker.init(frame, bbox)
    
#     # Display the resulting frame
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the reference image and extract keypoints and descriptors
ref_img = cv2.imread('left005200.png')
gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
ref_kp, ref_des = sift.detectAndCompute(gray, None)

# Set up the feature matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Set up the video capture
cap = cv2.VideoCapture('video.avi')

# Initialize the object bounding box
bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale and extract keypoints and descriptors
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # Match the keypoints and descriptors with those from the reference image
    matches = bf.match(ref_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract the matched keypoints
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    H, mask = cv2.findHomography(ref_pts, pts, cv2.RANSAC, 5.0)

    # Project the bounding box of the object from the reference image onto the current frame
    # if bbox is not None:
    #     bbox = cv2.perspectiveTransform(np.array([bbox]), H)[0]

    #     # Draw the bounding box on the current frame
    #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    # Project the bounding box of the object from the reference image onto the current frame
    if bbox is not None:
        bbox_pts = np.float32([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], 
                            [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]).reshape(-1, 1, 2)
        bbox_pts = cv2.perspectiveTransform(bbox_pts, H)
        bbox = cv2.boundingRect(bbox_pts)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    else:
        # If this is the first frame, ask the user to select the object bounding box
        bbox = cv2.selectROI(frame, False)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
