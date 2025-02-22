import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import os  # type: ignore

# Define CNN Model with customizable parameters
def build_custom_cnn(input_shape=(128, 128, 3), num_classes=29):
    """
    Builds a simple Convolutional Neural Network (CNN) for ASL classification.

    Try modifying the parameters below and see how training performance changes!

    Notice the input shape:

    128 x 128 x 3:
    
    ie: 
        Three 128 by 128 matrices - One for R, G, B


    """
    
    model = Sequential([
        # Convolutional Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),  # Try changing 32 to 16, 64, etc.
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),  # Experiment with different pooling sizes (2,2), (3,3)

        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Change 64 to 32, 128, etc.
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Fully Connected Layers
        Flatten(),
        Dense(256, activation='relu'),  # Try changing 256 to 128, 512, etc.
        Dropout(0.5),  # Try reducing Dropout (e.g., 0.3) to see if it affects overfitting
        Dense(num_classes, activation='softmax')  # Output layer (29 classes for ASL)
    ])

    # Try experimenting with different optimizers: 'adam', 'sgd', 'rmsprop'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


if os.path.exists('../saved_models/custom_asl_cnn.h5'):
    print("Loading saved model ...")
    model = tf.keras.models.load_model('../saved_models/custom_asl_cnn.h5')
else:
    print("No saved model. Training from scratch ...")
    model = build_custom_cnn()

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        'asl_alphabet_train/', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')

    val_generator = train_datagen.flow_from_directory(
        'asl_alphabet_train/', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

    # Train Model
    model.fit(train_generator, validation_data=val_generator, epochs=10)  # Try changing epochs to 5, 20, etc.

    # Save Model
    model.save('../saved_models/custom_asl_cnn.h5')
    print("Model saved to {model_path}")
