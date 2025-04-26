# pyright: reportMissingImports=false
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt

# Updated path to match first script
train_path = r"C:\Users\udupa\OneDrive\Desktop\gesture\train"

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # 20% validation from training set
)

train_batches = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(64, 64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True,
    subset='training'
)

valid_batches = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(64, 64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True,
    subset='validation'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='valid'),
    BatchNormalization(),
    MaxPool2D((2,2)),

    Dropout(0.2),
    Flatten(),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(len(train_batches.class_indices), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001),
    EarlyStopping(monitor='val_loss', patience=3),
    ModelCheckpoint('best_gesture_model.keras', monitor='val_accuracy', save_best_only=True)

]

# Train the model
history = model.fit(
    train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=15,
    callbacks=callbacks
)

# Save the final model
model.save('gesture_recognition_model.keras')


# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('training_history.png')
plt.show()