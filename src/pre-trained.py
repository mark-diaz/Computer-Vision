import tensorflow as tf # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

import numpy as np # type: ignore
import cv2 # type: ignore

# Load pre-trained MobileNetV2 (without the top classification layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model layers (to use as feature extractor)
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Flatten feature maps
x = Dense(128, activation='relu')(x)
x = Dense(29, activation='softmax')(x)  # 26 ASL letters

# Define final model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()  # Print model architecture

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'asl_alphabet_train/', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')

val_generator = train_datagen.flow_from_directory(
    'asl_alphabet_train/', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

history = model.fit(train_generator, validation_data=val_generator, epochs=1)

# # Load an image (replace 'test.jpg' with your own ASL image)
# img = cv2.imread('test.jpg')
# img = cv2.resize(img, (128, 128))
# img = img / 255.0  # Normalize
# img = np.expand_dims(img, axis=0)  # Add batch dimension

# # Predict
# pred = model.predict(img)
# class_idx = np.argmax(pred)
# print(f'Predicted ASL Letter: {chr(65 + class_idx)}')  # Convert to letter
