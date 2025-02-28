from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics           import confusion_matrix, classification_report

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, inception_v3
from tensorflow.keras import layers, models, callbacks, optimizers

# --- 1) PREPROCESS to 299x299 + inception_v3.preprocess_input---
def preprocess_inc(img):
    img299 = cv2.resize(img, (299, 299))
    # if grayscale, make 3-channel
    if img299.ndim == 2:
        img299 = np.stack([img299]*3, axis=-1)
    return inception_v3.preprocess_input(img299.astype(np.float32))

# --- 1a) BUILD X_inc, y_inc using preprocess_inc ---
X_inc = []
y_inc = []

for label, pairs in data.items():
    for img, _ in pairs:
        X_inc.append(preprocess_inc(img))
        y_inc.append(label_map[label])

X_inc = np.array(X_inc, dtype=np.float32)   # shape (N, 299, 299, 3)
y_inc = np.array(y_inc, dtype=np.int32)     # shape (N,)

# --- 2) SPLIT ---
X_trval, X_test, y_trval, y_test = train_test_split(
    X_inc, y_inc, test_size=0.15, stratify=y_inc, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=42
)

# --- 3) CLASS WEIGHTS ---
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))

# --- 4) AUGMENT + GENERATORS ---
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=None  # already preprocessed above
)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
val_gen   = val_datagen.flow(  X_val,   y_val,   batch_size=32)


# --- 5) BUILD & COMPILE ---
local_weights = "/kaggle/input/inceptionv3/tensorflow2/default/1/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
base = InceptionV3(weights=local_weights, include_top=False, input_shape=(299,299,3))
for layer in base.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(3, activation='softmax')(x)

model = models.Model(base.input, out)
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 6) CALLBACKS ---
es  = callbacks.EarlyStopping(    monitor='val_loss', patience=5, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# --- 7) TRAIN HEAD ---
history_head = model.fit(train_gen, epochs=20, validation_data=val_gen,
          class_weight=class_weight, callbacks=[es, rlp])

# --- 8) FINE-TUNE MOST LAYERS ---
for layer in base.layers:
    layer.trainable = True

model.compile(optimizer=optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_ft = model.fit(train_gen, epochs=20, validation_data=val_gen,
          class_weight=class_weight, callbacks=[es, rlp])

# --- 9) EVAL & REPORT ---
test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False)

loss, acc = model.evaluate(test_gen)
print("Test Acc:", acc)

preds = np.argmax(model.predict(test_gen), axis=1)
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=["Benign","Malignant","Normal"]))

import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix

# --- after your evaluation step ---
# preds = np.argmax(model.predict(test_gen), axis=1)
# loss, acc = model.evaluate(test_gen)
# y_test is your ground-truth label array

# 1) Compute confusion matrix
cm = confusion_matrix(y_test, preds)

# 2) Combine your two History objects (head & fine-tune) if you ran model.fit twice:
# history_head = model.fit(…)
# history_ft   = model.fit(…)
acc      = history_head.history['accuracy']    + history_ft.history['accuracy']
val_acc  = history_head.history['val_accuracy']+ history_ft.history['val_accuracy']
losses   = history_head.history['loss']        + history_ft.history['loss']
val_losses = history_head.history['val_loss']  + history_ft.history['val_loss']
epochs   = range(1, len(acc) + 1)

# 3) Plot in a 1×2 faceted layout
fig, (ax_cm, ax_perf) = plt.subplots(1, 2, figsize=(14, 6))

# — Confusion Matrix —
im = ax_cm.imshow(cm, interpolation='nearest')
ax_cm.set_title('InceptionV3 Confusion Matrix')
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha='center', va='center')
fig.colorbar(im, ax=ax_cm)

# — Performance Curves —
ax_perf.plot(epochs,      acc,        linestyle='-',  marker='o', label='Train Acc')
ax_perf.plot(epochs,      val_acc,    linestyle='--', marker='x', label='Val Acc')
ax_perf.plot(epochs,      losses,     linestyle=':',  marker='s', label='Train Loss')
ax_perf.plot(epochs,      val_losses, linestyle='-.', marker='d', label='Val Loss')
ax_perf.set_title('InceptionV3 Training History')
ax_perf.set_xlabel('Epoch')
ax_perf.legend(loc='best')
ax_perf.grid(True)

plt.tight_layout()
plt.savefig('inceptionv3_results.png', dpi=300)
plt.show()

# 4) Save test‐set metrics for later comparison
# Re-evaluate or reuse evaluation in fresh variables:
eval_loss, eval_acc = model.evaluate(test_gen, verbose=0)

eval_metrics = {
    'InceptionV3': {
        'loss': float(eval_loss),
        'accuracy': float(eval_acc)
    }
}

with open('inceptionv3_evaluation.json', 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print("Saved plot to inceptionv3_results.png")
print("Saved evaluation to inceptionv3_evaluation.json")
