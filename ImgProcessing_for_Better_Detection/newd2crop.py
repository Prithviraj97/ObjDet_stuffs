import cv2

# Read image
img = cv2.imread('left004441.png')

# Get person detection coordinates
x1, y1, x2, y2 = 972, 320, 1177, 694  # example detection coordinates

# Get center point of detection
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2

# Calculate crop coordinates
half_crop_size = 320  # half of desired crop size
crop_x1 = max(0, center_x - half_crop_size)
crop_y1 = max(0, center_y - half_crop_size)
crop_x2 = min(img.shape[1], center_x + half_crop_size)
crop_y2 = min(img.shape[0], center_y + half_crop_size)

# Crop image
crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

# Resize image to 640x640
resized_img = cv2.resize(crop_img, (640, 640))

# Display cropped and resized image
# cv2.imshow('Cropped and Resized Image', resized_img)
cv2.imwrite("cropleft4441New.png", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
