import cv2
import numpy as np

# Enable webcam to capture video
cap = cv2.VideoCapture(0)

# Define the codec for video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object to save the video
output = cv2.VideoWriter('captured_video.mov', fourcc, 20.0, (640, 480))

# Get time and frames, make capture continuous
while cap.isOpened():
    # Read frames from the camera
    ret, frame = cap.read()

    if ret:
        # Display the original image
        cv2.imshow('Input_frame', frame)

        # Write the frame to the output video file
        output.write(frame)

        # Use canny edge detector
        edges = cv2.Canny(frame, 200, 200)

        # Show edge detection
        cv2.imshow("Edges", edges)

        # Assign the kill key as Esc (27)
        if cv2.waitKey(1) == 27:
            break
    else:
        break

# Release the VideoWriter and capture resources
cap.release()
output.release()
cv2.destroyAllWindows()
