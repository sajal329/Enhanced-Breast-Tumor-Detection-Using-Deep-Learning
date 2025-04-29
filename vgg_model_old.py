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

################################################################################
# 7. COMPARE RESULTS
################################################################################

print("\nFinal Results:")
print(f"VGG16 Test Accuracy: {test_acc_vgg16:.4f}")
print(f"VGG19 Test Accuracy: {test_acc_vgg19:.4f}")

################################################################################
# 8. OPTIONAL: CONFUSION MATRIX & CLASSIFICATION REPORT
################################################################################
from sklearn.metrics import confusion_matrix, classification_report

# Predict classes for VGG16
preds_vgg16 = vgg16_model.predict(test_generator)
pred_classes_vgg16 = np.argmax(preds_vgg16, axis=1)
true_classes = y_test  # Use the original test labels

cm_vgg16 = confusion_matrix(true_classes, pred_classes_vgg16)
print("\n=== VGG16 Confusion Matrix ===")
print(cm_vgg16)

print("\n=== VGG16 Classification Report ===")
print(classification_report(true_classes, pred_classes_vgg16, target_names=["Benign", "Malignant", "Normal"]))

# Predict classes for VGG19
preds_vgg19 = vgg19_model.predict(test_generator)
pred_classes_vgg19 = np.argmax(preds_vgg19, axis=1)

cm_vgg19 = confusion_matrix(true_classes, pred_classes_vgg19)
print("\n=== VGG19 Confusion Matrix ===")
print(cm_vgg19)

print("\n=== VGG19 Classification Report ===")
print(classification_report(true_classes, pred_classes_vgg19, target_names=["Benign", "Malignant", "Normal"]))

import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix

# --- VGG16: Visualization & Save ---
# 1) Evaluate
eval_loss16, eval_acc16 = vgg16_model.evaluate(test_generator, verbose=0)

# 2) Predictions & Confusion Matrix
preds16 = np.argmax(vgg16_model.predict(test_generator), axis=1)
cm16    = confusion_matrix(y_test, preds16)

# 3) Training History
epochs16     = range(1, len(history_vgg16.history['accuracy']) + 1)
train_acc16  = history_vgg16.history['accuracy']
val_acc16    = history_vgg16.history['val_accuracy']
train_loss16 = history_vgg16.history['loss']
val_loss16   = history_vgg16.history['val_loss']

# 4) Plot 1×2 faceted figure
fig16, (ax_cm16, ax_perf16) = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
im16 = ax_cm16.imshow(cm16, interpolation='nearest', cmap='Blues')
ax_cm16.set_title('VGG16 Confusion Matrix')
ax_cm16.set_xlabel('Predicted')
ax_cm16.set_ylabel('True')
for i in range(cm16.shape[0]):
    for j in range(cm16.shape[1]):
        ax_cm16.text(j, i, cm16[i, j], ha='center', va='center')
fig16.colorbar(im16, ax=ax_cm16)

# Performance Curves
ax_perf16.plot(epochs16, train_acc16,   '-o', label='Train Acc')
ax_perf16.plot(epochs16, val_acc16,     '--x', label='Val Acc')
ax_perf16.plot(epochs16, train_loss16,  ':s', label='Train Loss')
ax_perf16.plot(epochs16, val_loss16,    '-.d', label='Val Loss')
ax_perf16.set_title('VGG16 Training History')
ax_perf16.set_xlabel('Epoch')
ax_perf16.legend(loc='best')
ax_perf16.grid(True)

plt.tight_layout()
plt.savefig('vgg16_results.png', dpi=300)
plt.show()

# 5) Save VGG16 metrics
vgg16_metrics = {
    'VGG16': {
        'loss': float(eval_loss16),
        'accuracy': float(eval_acc16)
    }
}
with open('vgg16_evaluation.json', 'w') as f:
    json.dump(vgg16_metrics, f, indent=2)

print("Saved VGG16 plot to vgg16_results.png")
print("Saved VGG16 evaluation to vgg16_evaluation.json")



# --- VGG19: Visualization & Save ---
# 1) Evaluate
eval_loss19, eval_acc19 = vgg19_model.evaluate(test_generator, verbose=0)

# 2) Predictions & Confusion Matrix
preds19 = np.argmax(vgg19_model.predict(test_generator), axis=1)
cm19    = confusion_matrix(y_test, preds19)

# 3) Training History
epochs19     = range(1, len(history_vgg19.history['accuracy']) + 1)
train_acc19  = history_vgg19.history['accuracy']
val_acc19    = history_vgg19.history['val_accuracy']
train_loss19 = history_vgg19.history['loss']
val_loss19   = history_vgg19.history['val_loss']

# 4) Plot 1×2 faceted figure
fig19, (ax_cm19, ax_perf19) = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
im19 = ax_cm19.imshow(cm19, interpolation='nearest', cmap='Blues')
ax_cm19.set_title('VGG19 Confusion Matrix')
ax_cm19.set_xlabel('Predicted')
ax_cm19.set_ylabel('True')
for i in range(cm19.shape[0]):
    for j in range(cm19.shape[1]):
        ax_cm19.text(j, i, cm19[i, j], ha='center', va='center')
fig19.colorbar(im19, ax=ax_cm19)

# Performance Curves
ax_perf19.plot(epochs19, train_acc19,   '-o', label='Train Acc')
ax_perf19.plot(epochs19, val_acc19,     '--x', label='Val Acc')
ax_perf19.plot(epochs19, train_loss19,  ':s', label='Train Loss')
ax_perf19.plot(epochs19, val_loss19,    '-.d', label='Val Loss')
ax_perf19.set_title('VGG19 Training History')
ax_perf19.set_xlabel('Epoch')
ax_perf19.legend(loc='best')
ax_perf19.grid(True)

plt.tight_layout()
plt.savefig('vgg19_results.png', dpi=300)
plt.show()

# 5) Save VGG19 metrics
vgg19_metrics = {
    'VGG19': {
        'loss': float(eval_loss19),
        'accuracy': float(eval_acc19)
    }
}
with open('vgg19_evaluation.json', 'w') as f:
    json.dump(vgg19_metrics, f, indent=2)

print("Saved VGG19 plot to vgg19_results.png")
print("Saved VGG19 evaluation to vgg19_evaluation.json")

