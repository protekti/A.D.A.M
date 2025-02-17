import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
data_path = '/root/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set'
label_file = '/root/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set/label_data_0601.json'

with open(label_file, 'r') as f:
    labels = [json.loads(line) for line in f]

# Function to preprocess images and labels
def preprocess_data(data_path, labels, img_size=(256, 256)):
    images = []
    masks = []
    for label in labels:
        img_path = os.path.join(data_path, label['raw_file'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        images.append(img)

        mask = np.zeros((720, 1280), dtype=np.uint8)
        lanes = label['lanes']
        h_samples = label['h_samples']
        for lane in lanes:
            if len(lane) > 0:
                points = [(x, y) for (x, y) in zip(lane, h_samples) if x >= 0]
                for i in range(len(points) - 1):
                    cv2.line(mask, points[i], points[i+1], 255, 5)
        mask = cv2.resize(mask, img_size)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    return images, masks
    
# Preprocess the data
images, masks = preprocess_data(data_path, labels)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Normalize the images and masks
x_train = x_train / 255.0
x_val = x_val / 255.0
y_train = y_train / 255.0
y_val = y_val / 255.0

# Expand dimensions of masks to match model output shape
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    
    # Decoder
    up1 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv4)
    concat1 = layers.concatenate([up1, conv3])
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    
    up2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv5)
    concat2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    
    up3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv6)
    concat3 = layers.concatenate([up3, conv1])
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat3)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv7)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
input_shape = (256, 256, 3)
model = build_model(input_shape)
model.summary()

# Callback to save the best model automatically
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[checkpoint], verbose=1)

# Save the final model
model.save("final_model.h5")

# Evaluate the model
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=2)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")