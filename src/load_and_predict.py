import numpy as np # type: ignore
import cv2 # type: ignore
import tensorflow as tf # type: ignore

def load_and_predict(model, image_path):
    """Loads an image, preprocesses it, and predicts the ASL letter."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Could not read image {image_path}")

    # Resize and normalize the image
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Make a prediction
    pred = model.predict(img_expanded)
    class_idx = np.argmax(pred)
    predicted_letter = chr(65 + class_idx)
    print(f'Predicted ASL Letter: {predicted_letter}')

    # Show image with prediction
    cv2.imshow("ASL Letter", img_resized)

    # Wait for user input with timeout
    while True:
        key = cv2.waitKey(10) & 0xFF  # Check every 10ms
        if key == ord('q') or key == 27:  # Press 'q' or 'Esc' to close
            break

    cv2.destroyAllWindows()
    
    return predicted_letter
