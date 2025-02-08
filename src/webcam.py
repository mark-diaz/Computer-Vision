import cv2  # type: ignore

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the rectangle's position and size
rect_width = 100
rect_height = 100
rect_x = (frame_width - rect_width) // 2
rect_y = (frame_height - rect_height) // 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Draw a rectangle in the center
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
