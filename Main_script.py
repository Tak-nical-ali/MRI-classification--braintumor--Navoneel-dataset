import os
import tensorflow as tf 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot

base_dir ='C:\ML training\Brain_MRI_classification\data\Brain_tumor' 

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VAL_SPLIT = 0.2

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, 
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', 
    validation_split=VAL_SPLIT)

train_generator = train_datagen.flow_from_directory(
    base_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='training', seed=42
)

validation_generator = train_datagen.flow_from_directory(
    base_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation', seed=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    # Layer 1: Convolutional + Pooling
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),

    # Layer 2: Convolutional + Pooling
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Layer 3: Convolutional + Pooling (Optional, sometimes omitted for smaller datasets)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten the 3D output to 1D for the Dense layers
    Flatten(),

    # Dropout for regularization
    Dropout(0.5),

    # Dense (Fully Connected) Layers
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])



from tensorflow.keras.optimizers import Adam
model.compile(optimizer= Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])



EPOCHS = 15 # Start with a reasonable number of epochs

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE)


import matplotlib.pyplot as plt

# Plot training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# You can also evaluate directly on the validation set for final metrics
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")