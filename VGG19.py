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
