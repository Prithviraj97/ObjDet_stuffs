import cv2
import numpy as np
img = cv2.imread('left005200.png')

circles = []
def circle_tool(event, x, y, flags, params):
    global img,color,circles
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a circle with the current position as the center
        cv2.circle(img, (x, y), radius, color, thickness)
        circles.append((x, y, radius, color, thickness))
        
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        # Draw a circle with the current position as the center
        cv2.circle(img, (x, y), radius, color, thickness)
        
    elif event == cv2.EVENT_MBUTTONDOWN:
        # Change the color
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
# def undo_circle(event, x, y, flags, params):
#     global image, circles
    
#     if event == cv2.EVENT_RBUTTONDOWN:
#         if circles:
#             # Remove the last circle parameters from the list
#             circles.pop()
            
#             # Reset the image
#             image = np.copy(img)
            
#             # Draw all the remaining circles
#             for circle in circles:
#                 x, y, r, c, t = circle
#                 cv2.circle(img, (x, y), r, c, t)


radius = 5
x,y = 20, 20
color = (0, 0, 255)
thickness = -1

cv2.namedWindow('circle_tool')
cv2.setMouseCallback('circle_tool', circle_tool)
# cv2.setMouseCallback('circle_tool', undo_circle, param=image)

while True:
    # Display the image
    cv2.imshow('circle_tool', img)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Exit loop if 'q' is pressed
    if key == ord('q'):
        break
    
    if key == ord('s'):
        cv2.imwrite('left005200wCircel.png', img)

# Clean up
cv2.destroyAllWindows()
