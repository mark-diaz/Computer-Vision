import tensorflow as tf # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os # type: ignore

import numpy as np # type: ignore
import cv2 # type: ignore

from load_and_predict import load_and_predict  # Import the function



if os.path.exists('../saved_models/asl_model.h5'):
    print("Loading saved model ...")
    model = load_model('../saved_models/asl_model.h5')  # Load the saved model
else:
    print("No saved model training from scratch ...")

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
    
    model.save('../saved_models/asl_model.h5')
    print("Model saved!")

# Let's see how they test:

# Test with different ASL images
load_and_predict(model, 'asl_alphabet_test/A_test.jpg')
load_and_predict(model, 'asl_alphabet_test/B_test.jpg')
load_and_predict(model, 'asl_alphabet_test/Z_test.jpg')