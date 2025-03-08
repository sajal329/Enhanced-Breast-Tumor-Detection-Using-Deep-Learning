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

