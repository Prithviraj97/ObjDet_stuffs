import cv2
import numpy as np

sift = cv2.SIFT_create()
ref_img = cv2.imread('left005200.png', cv2.IMREAD_GRAYSCALE)
ref_kp, ref_des = sift.detectAndCompute(ref_img, None)

cap = cv2.VideoCapture('video.avi')
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

tracker = cv2.TrackerKCF_create()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    
    matches = bf.match(ref_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract the matched keypoints
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate the homography matrix
    H, mask = cv2.findHomography(ref_pts, pts, cv2.RANSAC, 5.0)
    
    # Use the homography matrix to get the bounding box of the object in the current frame
    h, w = ref_img.shape
    bbox = cv2.perspectiveTransform(np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32).reshape(-1, 1, 2), H)
    bbox = bbox.reshape(-1, 2).astype(int)
    
    # Update the tracker with the bounding box
    ok = tracker.init(frame, tuple(bbox))
    if ok:
        ok, bbox = tracker.update(frame)
    
    # Draw the bounding box on the frame
    if ok:
        bbox = tuple(map(int, bbox))
        cv2.rectangle(frame, bbox[0:2], bbox[2:4], (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
