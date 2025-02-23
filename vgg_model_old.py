import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# For splitting data
from sklearn.model_selection import train_test_split

# For data augmentation and model building in Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras import layers, models, optimizers

################################################################################
# 1. LOAD & PREPROCESS DATA
################################################################################

# Suppose you have a dictionary named `data` with keys: 'benign', 'malignant', 'normal'.
# Each entry is a list of (image, mask) tuples, but for classification, we only need the image.
# Example: data['benign'] = [(img1, mask1), (img2, mask2), ... ]

# Map each label to a numeric class (0, 1, 2). Adjust if you prefer a different order.
label_map = {
    'benign': 0,
    'malignant': 1,
    'normal': 2
}

def preprocess_image(img, size=(224, 224)):
    """
    1) Resize to `size`.
    2) Normalize to [0,1].
    3) Convert grayscale -> 3 channels for VGG (if needed).
    """
    # Resize
    img_resized = cv2.resize(img, size)

    # Normalize
    img_resized = img_resized.astype(np.float32) / 255.0

    # If single-channel (H,W), expand to (H,W,1) then tile to (H,W,3)
    if len(img_resized.shape) == 2:  # grayscale
        img_resized = np.expand_dims(img_resized, axis=-1)  # (H, W, 1)
        img_resized = np.tile(img_resized, (1, 1, 3))       # (H, W, 3)

    return img_resized

# Gather images (X) and labels (y)
X = []
y = []

for label, pairs in data.items():
    for (img, mask) in pairs:
        processed_img = preprocess_image(img, size=(224, 224))
        X.append(processed_img)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape, "Labels shape:", y.shape)
# Example: (780, 224, 224, 3)  (780,)

################################################################################
# 2. SPLIT DATA INTO TRAIN, VAL, TEST
################################################################################

# First, separate out a test set (15%), then a validation set (15% of the remainder)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)

print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Test size:", len(X_test))

################################################################################
# 3. DATA AUGMENTATION
################################################################################

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation and test, we typically only do rescaling (already normalized in preprocess)
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True
)

val_generator = val_datagen.flow(
    X_val, y_val,
    batch_size=batch_size,
    shuffle=False
)

test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=batch_size,
    shuffle=False
)
