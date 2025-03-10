import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# REPRODUCIBILITY SETUP
# -------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CUDNN_DETERMINISM'] = '1'

# --- SETTINGS ---
BASE_PATH = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 3  # benign, malignant, normal

# --- LABEL MAP ---
LABELS = {'benign': 0, 'malignant': 1, 'normal': 2}

# --- DATA LOADING & PREPROCESSING ---
def load_images(base_path):
    X, y = [], []
    for label, idx in LABELS.items():
        folder = os.path.join(base_path, label)
        files = [f for f in os.listdir(folder) if f.endswith('.png') and '_mask' not in f]
        for f in files:
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, -1)
            img = np.tile(img, (1, 1, 3))  # to 3-channel
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

# Load data
X, y = load_images(BASE_PATH)
print(f"Loaded {len(X)} images with shape {X.shape[1:]}.")

# Split into train/val/test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Data augmentation
gen_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
).flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
gen_val = ImageDataGenerator().flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# Model builder
def build_model(input_shape, num_classes):
    base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers[:-4]:  # freeze except last 4
        layer.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return models.Model(inputs=base.input, outputs=outputs)

# Build and compile
model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Callbacks
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    callbacks.ModelCheckpoint('best_vgg16.keras', save_best_only=True)
]

# Train
history = model.fit(
    gen_train,
    epochs=EPOCHS,
    validation_data=gen_val,
    callbacks=cb
)

# Evaluate on test set
gen_test = ImageDataGenerator().flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
loss, acc = model.evaluate(gen_test)
print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# Predict & metrics
y_pred = np.argmax(model.predict(gen_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=list(LABELS.keys())))

# Plots
epochs_range = range(1, len(history.history['loss']) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], '-o', label='Train Acc')
plt.plot(epochs_range, history.history['val_accuracy'], '--x', label='Val Acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], '-o', label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], '--x', label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout()
plt.savefig('performance_16.png', dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.colorbar(); plt.tight_layout()
plt.savefig('confusion_matrix_16.png', dpi=300)
plt.show()

# -------------------------
# SAVE MATRICES ##
# -------------------------
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report

# 1) Compute predictions and metrics
y_pred = np.argmax(model.predict(gen_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test, 
    y_pred, 
    target_names=list(LABELS.keys()), 
    output_dict=True
)

# 2) Save to disk
#   - confusion matrix as a .npy
#   - both cm and report in one JSON
np.save('confusion_matrix_vgg16.npy', cm)

with open('results_vgg16.json', 'w') as f:
    json.dump({
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }, f, indent=2)

print("Saved cm → confusion_matrix.npy and values → results.json")
