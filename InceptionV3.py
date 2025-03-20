import os, random, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, inception_v3
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow as tf

# -------------------------
# 1) REPRODUCIBILITY SETUP
# -------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

tf.random.set_seed(SEED)
# Use cuDNN deterministic if available
os.environ['TF_CUDNN_DETERMINISM'] = '1'

# -------------------------
# 2) SETTINGS
# -------------------------
BASE_PATH    = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
IMAGE_SIZE   = (299, 299)
BATCH_SIZE   = 16
EPOCHS_HEAD  = 15
EPOCHS_TUNE  = 30
NUM_CLASSES  = 3

LABEL_MAP = {'benign': 0, 'malignant': 1, 'normal': 2}

# -------------------------
# 3) DATA LOADING + PREPROCESS
# -------------------------
def load_and_preprocess(path):
    X, y = [], []
    for label, idx in LABEL_MAP.items():
        folder = os.path.join(path, label)
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if not fname.endswith('.png') or '_mask' in fname: continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, IMAGE_SIZE)
            # replicate channels
            img = np.stack([img, img, img], axis=-1)
            X.append(inception_v3.preprocess_input(img.astype(np.float32)))
            y.append(idx)
    return np.array(X), np.array(y)

X, y = load_and_preprocess(BASE_PATH)
print(f"Loaded data: {X.shape}, Labels: {np.unique(y, return_counts=True)}")

# -------------------------
# 4) SPLIT
# -------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
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
    brightness_range=[0.7, 1.3]
)
val_aug = ImageDataGenerator()
test_aug = ImageDataGenerator()

gen_train = train_aug.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
gen_val   = val_aug.flow(X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)
gen_test  = test_aug.flow(X_test,  y_test,  batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 7) BUILD MODEL
# -------------------------
def build_inception(input_shape, num_classes):
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    # freeze all layers initially
    for layer in base.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model

model = build_inception((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# -------------------------
# 8) CALLBACKS
# -------------------------
callbacks_list = [
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
    callbacks=callbacks_list
)

# -------------------------
# 10) FINE-TUNE LAST INCEPTION BLOCK
# -------------------------
# unfreeze last 2 inception blocks
for layer in model.layers:
    if 'mixed10' in layer.name or 'mixed9' in layer.name:
        layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history_ft = model.fit(
    gen_train,
    epochs=EPOCHS_TUNE,
    validation_data=gen_val,
    class_weight=class_weight,
    callbacks=callbacks_list
)

# -------------------------
# 11) EVALUATION
# -------------------------
loss, acc = model.evaluate(gen_test)
print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")

y_pred = np.argmax(model.predict(gen_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=list(LABEL_MAP.keys())))
