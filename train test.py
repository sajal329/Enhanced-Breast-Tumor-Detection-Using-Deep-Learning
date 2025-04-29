import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(img, size=(224, 224)):
    """
    Resizes the image to `size` and normalizes pixel values to [0,1].
    For grayscale images, expand dims to get shape (H, W, 1).
    """
    # Resize to the desired size
    img_resized = cv2.resize(img, size)

    # Convert to float and normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # If it's a single-channel image, expand dims to shape (224,224,1)
    if len(img_normalized.shape) == 2:
        img_normalized = np.expand_dims(img_normalized, axis=-1)

    return img_normalized


# Example label mapping
label_map = {
    'benign': 0,
    'malignant': 1,
    'normal': 2
}

X = []
y = []

# Iterate over each class label and its (image, mask) pairs
for label, pairs in data.items():
    for (img, mask) in pairs:
        # Preprocess the image
        processed_img = preprocess_image(img, size=(224, 224))
        
        X.append(processed_img)
        y.append(label_map[label])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (num_samples, 224, 224, 1) or (num_samples, 224, 224, 3)
print("y shape:", y.shape)  # (num_samples,)


# 1) Split train+val vs test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# 2) Split train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.15,  # 15% of (train+val)
    stratify=y_trainval,
    random_state=42
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))
