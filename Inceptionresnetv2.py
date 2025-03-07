import numpy as np
import cv2
from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics           import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications        import InceptionResNetV2, inception_resnet_v2
from tensorflow.keras import layers, models, optimizers, callbacks

# 1) PREPROCESS FUNCTION FOR InceptionResNetV2 (299×299 + 3-channel + correct normalization)
def preprocess_ir2(img):
    # Resize from original (e.g. 500×500) → 299×299
    img299 = cv2.resize(img, (299, 299))
    # Grayscale → 3-channel
    if img299.ndim == 2:
        img299 = np.stack([img299]*3, axis=-1)
    # Apply InceptionResNetV2’s own preprocessing
    return inception_resnet_v2.preprocess_input(img299.astype(np.float32))

# 2) BUILD THE DATA ARRAYS
label_map = {'benign': 0, 'malignant': 1, 'normal': 2}
X_ir2, y_ir2 = [], []
for label, pairs in data.items():
    for img, _ in pairs:
        X_ir2.append(preprocess_ir2(img))
        y_ir2.append(label_map[label])
X_ir2 = np.array(X_ir2)
y_ir2 = np.array(y_ir2)

# 3) SPLIT INTO TRAIN / VAL / TEST
X_trval, X_test, y_trval, y_test = train_test_split(
    X_ir2, y_ir2, test_size=0.15, stratify=y_ir2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=42
)

# 4) COMPUTE CLASS WEIGHTS
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))

# 5) DATA AUGMENTATION & GENERATORS
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen  = ImageDataGenerator()
test_datagen = ImageDataGenerator()

batch_size = 32
train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
val_gen   = val_datagen.flow(  X_val,   y_val,   batch_size=batch_size, shuffle=False)
test_gen  = test_datagen.flow( X_test,  y_test,  batch_size=batch_size, shuffle=False)

