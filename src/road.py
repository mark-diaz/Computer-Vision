import numpy as np # type: ignore
import cv2 # type: ignore

while True:
    # Create a blank image of size 300x300 with 3 color channels (initialized to black)
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Set all pixels to red
    image[0:255, 0:255] = [0, 0, 255]  # Blue, Green, Red

    # Display the image
    cv2.imshow('Red Image', image)


    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
