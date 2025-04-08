import os, random, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2, inception_resnet_v2
from tensorflow.keras import layers, models, optimizers, callbacks

# -------------------------
# 1) REPRODUCIBILITY SETUP
# -------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def set_tf_seed(seed=SEED):
    tf.random.set_seed(seed)
    os.environ['TF_CUDNN_DETERMINISM'] = '1'
set_tf_seed()

# -------------------------
# 2) SETTINGS
# -------------------------
BASE_PATH    = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
IMAGE_SIZE   = (299, 299)
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
            if img is None: continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = np.stack([img]*3, axis=-1)
            img = inception_resnet_v2.preprocess_input(img.astype(np.float32))
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

X, y = load_and_preprocess(BASE_PATH)
print(f"Loaded {X.shape[0]} samples, image shape {X.shape[1:]}.")

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
# 7) BUILD InceptionResNetV2 MODEL
# -------------------------
# Use imagenet weights
base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))
# Freeze all layers initially
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
cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
]

# -------------------------
# 9) TRAIN HEAD
# -------------------------
hist_head = model.fit(
    gen_train,
    epochs=EPOCHS_HEAD,
    validation_data=gen_val,
    class_weight=class_weight,
    callbacks=cb_list
)

# -------------------------
# 10) FINE-TUNE LAST BLOCKS
# -------------------------
# Unfreeze last 2 inception blocks
for layer in base.layers:
    if 'conv_7b' in layer.name or 'conv_6b' in layer.name:
        layer.trainable = True

# Recompile with lower LR
tf.keras.backend.clear_session()
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
set_tf_seed()
hist_ft = model.fit(
    gen_train,
    epochs=EPOCHS_TUNE,
    validation_data=gen_val,
    class_weight=class_weight,
    callbacks=cb_list
)

# -------------------------
# 11) EVALUATION
# -------------------------
eval_loss, eval_acc = model.evaluate(gen_test)
print(f"Test Loss: {eval_loss:.4f}, Test Acc: {eval_acc:.4f}")

preds = np.argmax(model.predict(gen_test), axis=1)
cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, preds, target_names=list(LABEL_MAP.keys())))

# -------------------------
# 12) PLOTTING
# -------------------------
train_acc   = hist_head.history['accuracy']    + hist_ft.history['accuracy']
val_acc     = hist_head.history['val_accuracy']+ hist_ft.history['val_accuracy']
train_loss  = hist_head.history['loss']        + hist_ft.history['loss']
val_loss    = hist_head.history['val_loss']    + hist_ft.history['val_loss']
epochs_all  = range(1, len(train_acc) + 1)

# --- Performance curves ---
plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(epochs_all, train_acc, '-o', label='Train Acc')
plt.plot(epochs_all, val_acc,   '--x', label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(epochs_all, train_loss, '-o', label='Train Loss')
plt.plot(epochs_all, val_loss,   '--x', label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('ir2_performance.png', dpi=300)
plt.show()


# --- Confusion matrix heatmap ---
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('InceptionResNetV2 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.tight_layout()
plt.savefig('ir2_confusion.png', dpi=300)
plt.show()

# -------------------------
# 13) SAVE METRICS
# -------------------------
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report

# 1) Compute predictions and metrics
preds = np.argmax(model.predict(gen_test), axis=1)
cm = confusion_matrix(y_test, preds)
report = classification_report(
    y_test, 
    preds, 
    target_names=list(LABELS.keys()), 
    output_dict=True
)

# 2) Save to disk
#   - confusion matrix as a .npy
#   - both cm and report in one JSON
np.save('confusion_matrix.npy', cm)

with open('results.json', 'w') as f:
    json.dump({
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }, f, indent=2)

print("Saved cm → confusion_matrix.npy and values → results.json")


import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Map model names to JSON filenames
model_files = {
    'VGG16': 'results_vgg16.json',
    'VGG19': 'results_vgg19.json',
    'InceptionV3': 'results_inceptionV3.json',
    'DenseNet169': 'results_densenet169.json',
    'InceptionResNetV2': 'results.json'
}

# 2) Load all results into a dict
results = {}
for model, fname in model_files.items():
    with open(fname, 'r') as f:
        results[model] = json.load(f)

# 3) Extract metrics
#   • Overall accuracy
accuracy = {m: r['classification_report']['accuracy']
            for m, r in results.items()}

#   • Macro and weighted averages
macro = pd.DataFrame({m: r['classification_report']['macro avg']
                      for m, r in results.items()}).T
weighted = pd.DataFrame({m: r['classification_report']['weighted avg']
                         for m, r in results.items()}).T

#   • Per-class F1 scores
classes = ['benign', 'malignant', 'normal']
class_f1 = pd.DataFrame({
    cls: {m: results[m]['classification_report'][cls]['f1-score']
          for m in results}
    for cls in classes
})

# 4) Line plot: Accuracy and Macro F1-score
plt.figure(figsize=(12, 5))
models = list(accuracy.keys())
acc_vals = [accuracy[m] for m in models]
f1_vals = [macro.loc[m, 'f1-score'] for m in models]

plt.plot(models, acc_vals, '-o', label='Accuracy')
plt.plot(models, f1_vals, '--x', label='Macro F1-score')
plt.title('Model Comparison: Accuracy vs. Macro F1-score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('comparison_accuracy_macrof1.png')
plt.show()

# 5) Bar chart: Weighted-average metrics
weighted[['precision', 'recall', 'f1-score']].plot(
    kind='bar', figsize=(10, 6))
plt.title('Weighted Average Metrics by Model')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('weighted_metrics_by_model.png')
plt.show()

# 6) Bar chart: Per-class F1-scores
class_f1.plot(kind='bar', figsize=(10, 6))
plt.title('Per-Class F1-score across Models')
plt.ylabel('F1-score')
plt.tight_layout()
plt.savefig('per_class_f1_by_model.png')
plt.show()

# 7) Confusion matrix heatmaps (2×3 grid)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (model, res) in zip(axes, results.items()):
    cm = np.array(res['confusion_matrix'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(model)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center')
# Turn off the unused subplot if any
if len(results) < len(axes):
    axes[-1].axis('off')

fig.colorbar(im, ax=axes.tolist())
plt.tight_layout()
plt.savefig('all_confusion_matrices.png')
plt.show()
