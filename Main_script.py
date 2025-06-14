import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot
import logging
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR) 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


base_dir = 'C:\ML training\Brain_MRI_classification\data\Brain_tumor'

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


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])


from tensorflow.keras.optimizers import SGD
model.compile(optimizer= SGD(), # Your current learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])


EPOCHS = 20
checkpoint_filepath = 'temporarily_best_brain_mri_model.h5'

# Create the ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,  # Set to True if you only want to save weights, False to save entire model
    monitor='val_accuracy',   # Metric to monitor for improvement
    mode='max',               # 'max' means we want to maximize 'val_accuracy'
    save_best_only=True,      # Only save when the monitored metric improves
    verbose=1                 # Show messages when a model is saved
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[model_checkpoint_callback] # Pass the callback list to model.fit
)

def error(y_test, y_preds):
  mae_fn= tf.keras.losses.MeanAbsoluteError()
  mae= mae_fn(y_test, y_preds).numpy()
  mse_fn = tf.keras.losses.MeanSquaredError()
  mse = mse_fn(y_test, y_preds).numpy()
  print(f"Mean absolute error is: {mae}")
  print(f"Mean squared error is: {mse}")

# Plotting code remains the same
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
# IMPORTANT: This evaluation will be on the model's state at the VERY LAST epoch,
# not necessarily the best epoch saved by the checkpoint.
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss (Last Epoch Model): {loss:.4f}")
print(f"Validation Accuracy (Last Epoch Model): {accuracy:.4f}")
