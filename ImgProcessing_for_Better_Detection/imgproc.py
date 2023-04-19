import cv2

# load the image
img = cv2.imread('left005200.png')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 0, 110, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# apply morphology operations to remove noise and fill holes in the ball
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# create a white background image
bg = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
bg.fill(255)

# mask the original image with the binary image and show the result
result = cv2.bitwise_and(img, bg, mask=closing)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
