import os, random, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow as tf

# -------------------------
# 1) REPRODUCIBILITY SETUP
# -------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def _set_tf_seed():
    tf.random.set_seed(SEED)
    try:
        os.environ['TF_CUDNN_DETERMINISM'] = '1'
    except:
        pass
_set_tf_seed()

# -------------------------
# 2) SETTINGS
# -------------------------
BASE_PATH    = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 16
EPOCHS_HEAD  = 15
EPOCHS_TUNE  = 30
NUM_CLASSES  = 3
LABEL_MAP    = {'benign': 0, 'malignant': 1, 'normal': 2}

# -------------------------
# 3) DATA LOADING + PREPROCESSING
# -------------------------
def load_and_preprocess(path):
    X, y = [], []
    for label, idx in LABEL_MAP.items():
        folder = os.path.join(path, label)
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if not fname.endswith('.png') or '_mask' in fname:
                continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = np.stack([img]*3, axis=-1)
            img = densenet_preprocess(img.astype(np.float32))
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

X, y = load_and_preprocess(BASE_PATH)
print(f"Loaded {X.shape[0]} samples with shape {X.shape[1:]}.")

# -------------------------
# 4) SPLIT DATA
# -------------------------
X_trval, X_test, y_trval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=SEED
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -------------------------
# 5) CLASS WEIGHTS
# -------------------------
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))
print(f"Class weights: {class_weight}")

# -------------------------
# 6) AUGMENTATION + GENERATORS
# -------------------------
train_aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7,1.3]
)
val_aug = ImageDataGenerator()
test_aug = ImageDataGenerator()

gen_train = train_aug.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
gen_val   = val_aug.flow(X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)
gen_test  = test_aug.flow(X_test,  y_test,  batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 7) BUILD DenseNet169 MODEL
# -------------------------
local_weights = "/kaggle/input/densenet169/tensorflow2/default/1/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5"
try:
    base = DenseNet169(weights='imagenet', include_top=False, input_shape=(224,224,3))
except:
    base = DenseNet169(weights=local_weights, include_top=False, input_shape=(224,224,3))
# Freeze base
for layer in base.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# -------------------------
# 8) CALLBACKS
# -------------------------
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
]

# -------------------------
# 9) TRAIN HEAD
# -------------------------
history_head = model.fit(
    gen_train,
    epochs=EPOCHS_HEAD,
    validation_data=gen_val,
    class_weight=class_weight,
    callbacks=cb
)
