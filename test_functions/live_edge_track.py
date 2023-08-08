import cv2
import numpy as np

# Enable webcam to capture video 
cap = cv2.VideoCapture(0)

d = 30
sigmaColor = 30
sigmaSpace = 20

t_lower = 5
t_upper = 10

# Get time and frames, make capture continuous
while(1):
    # read frames from camera
    ret, frame = cap.read()
    
    # Display original image
    # cv2.imshow('Input_frame', frame)

    # # get gray image
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Apply bilateral filtering
    # gray_filtered = cv2.bilateralFilter(gray,
    #                                     d=d,
    #                                     sigmaColor=sigmaColor,
    #                                     sigmaSpace=sigmaSpace)
    
    # # Detect edges with Canny
    # edges_filtered = cv2.Canny(gray_filtered,
    #                            threshold1=t_lower,
    #                            threshold2=t_upper)

    # Use canny edge detector 
    edges = cv2.Canny(frame, 100, 100)

    # Show edge detection
    cv2.imshow("Edges", edges)

    # Assign kill key as Esc (27)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close Window
cap.release()
cv2.destroyAllWindows()


#############################################################

# Live edge motion

# fgbg = cv2.createBackgroundSubtractorMOG2(
#     history=10,
#     varThreshold=2,
#     detectShadows=False)

# cap = cv2.VideoCapture(0)

# while(1):
#     # Read the frames in from the camera
#     ret, frame = cap.read()
#     if ret == True:
#         # convert to gray
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Extract foreground 
#         edges_foreground = cv2.bilateralFilter(gray, 9,75,75)
#         foreground = fgbg.apply(edges_foreground)

#         # Smooth out to get the moving area
#         kernel = np.ones((50,50),np.uint8)
#         foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

#         # Applying static edge extraction
#         edges_foreground = cv2.bilateralFilter(gray, 9, 100, 100)
#         edges_filtered = cv2.Canny(edges_foreground, 60, 120)

#         # Crop off the edges out of the moving area
#         cropped = (foreground // 255) * edges_filtered

#         # Stacking the images to print them together for comparison
#         images = np.hstack((edges_filtered, cropped))

#         # Show image
#         cv2.imshow("frames",images)

#         # Assign kill key
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()


