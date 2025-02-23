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

################################################################################
# 4. BUILD VGG16 & VGG19 MODELS (TRANSFER LEARNING)
################################################################################

def build_vgg16_model(input_shape=(224, 224, 3), num_classes=3, freeze_until=15):
    """
    Build a VGG16 model (pretrained on ImageNet) for multi-class classification.
    freeze_until: number of layers to freeze in the base model (set to None to unfreeze all).
    """
    local_weights_path = "/kaggle/input/vgg16/keras/default/1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    
    base_model = VGG16(
        weights=local_weights_path,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the first `freeze_until` layers
    if freeze_until is not None:
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
    
    # Build classification head
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def build_vgg19_model(input_shape=(224, 224, 3), num_classes=3, freeze_until=17):
    """
    Build a VGG19 model (pretrained on ImageNet) for multi-class classification.
    freeze_until: number of layers to freeze in the base model (set to None to unfreeze all).
    """
    local_weights_path = "/kaggle/input/vgg19/tensorflow2/default/1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    
    base_model = VGG19(
        weights=local_weights_path,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the first `freeze_until` layers
    if freeze_until is not None:
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
    
    # Build classification head
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

################################################################################
# 5. TRAIN & EVALUATE VGG16
################################################################################

vgg16_model = build_vgg16_model(input_shape=(224, 224, 3), num_classes=3, freeze_until=15)
vgg16_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n=== VGG16 Model Summary ===")
vgg16_model.summary()

# Train
history_vgg16 = vgg16_model.fit(
    train_generator,
    epochs=20,  # increase for better results
    validation_data=val_generator
)

# Evaluate on test set
test_loss_vgg16, test_acc_vgg16 = vgg16_model.evaluate(test_generator)
print("VGG16 Test Accuracy:", test_acc_vgg16)

################################################################################
# 6. TRAIN & EVALUATE VGG19
################################################################################

vgg19_model = build_vgg19_model(input_shape=(224, 224, 3), num_classes=3, freeze_until=17)
vgg19_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n=== VGG19 Model Summary ===")
vgg19_model.summary()

# Train
history_vgg19 = vgg19_model.fit(
    train_generator,
    epochs=20,  # increase for better results
    validation_data=val_generator
)

# Evaluate on test set
test_loss_vgg19, test_acc_vgg19 = vgg19_model.evaluate(test_generator)
print("VGG19 Test Accuracy:", test_acc_vgg19)

