import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG19
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
        for fname in os.listdir(folder):
            if not fname.endswith('.png') or '_mask' in fname:
                continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype('float32') / 255.0
            img = np.stack([img]*3, axis=-1)
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

# Load data
X, y = load_images(BASE_PATH)
print(f"Loaded {len(X)} images with shape {X.shape[1:]}.")

# --- SPLIT DATA ---
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- DATA AUGMENTATION ---
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

gen_train = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
gen_val   = val_datagen.flow(  X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)

# --- BUILD MODEL ---
def build_model(input_shape, num_classes):
    base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers[:-4]:
        layer.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return models.Model(inputs=base.input, outputs=outputs)

model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
