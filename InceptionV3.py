from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics           import confusion_matrix, classification_report

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, inception_v3
from tensorflow.keras import layers, models, callbacks, optimizers

# --- 1) PREPROCESS to 299x299 + inception_v3.preprocess_input---
def preprocess_inc(img):
    img299 = cv2.resize(img, (299, 299))
    # if grayscale, make 3-channel
    if img299.ndim == 2:
        img299 = np.stack([img299]*3, axis=-1)
    return inception_v3.preprocess_input(img299.astype(np.float32))

# --- 1a) BUILD X_inc, y_inc using preprocess_inc ---
X_inc = []
y_inc = []

for label, pairs in data.items():
    for img, _ in pairs:
        X_inc.append(preprocess_inc(img))
        y_inc.append(label_map[label])

X_inc = np.array(X_inc, dtype=np.float32)   # shape (N, 299, 299, 3)
y_inc = np.array(y_inc, dtype=np.int32)     # shape (N,)

# --- 2) SPLIT ---
X_trval, X_test, y_trval, y_test = train_test_split(
    X_inc, y_inc, test_size=0.15, stratify=y_inc, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=42
)

# --- 3) CLASS WEIGHTS ---
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))

# --- 4) AUGMENT + GENERATORS ---
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=None  # already preprocessed above
)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_gen   = val_datagen.flow(  X_val,   y_val,   batch_size=32)
