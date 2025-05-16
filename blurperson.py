import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to blur human bodies in an image
def blur_human_body(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    results = model(img)
    detections = results.xyxy[0].numpy()

    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if cls == 0:  
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            # Apply Gaussian blur to detected human body.
            roi_blurred = cv2.GaussianBlur(roi, (51, 51), 0)
            # Replace the original ROI with the blurred ROI in the image
            img[int(y1):int(y2), int(x1):int(x2)] = roi_blurred

    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Blurred image saved to {output_path}")

# Usage
image_path = 'G:\YoloFrames\left000310.png'
output_path = 'output_left000310.png'
blur_human_body(image_path, output_path)
