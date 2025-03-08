import numpy as np
import cv2
from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics           import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications        import InceptionResNetV2, inception_resnet_v2
from tensorflow.keras import layers, models, optimizers, callbacks

# 1) PREPROCESS FUNCTION FOR InceptionResNetV2 (299×299 + 3-channel + correct normalization)
def preprocess_ir2(img):
    # Resize from original (e.g. 500×500) → 299×299
    img299 = cv2.resize(img, (299, 299))
    # Grayscale → 3-channel
    if img299.ndim == 2:
        img299 = np.stack([img299]*3, axis=-1)
    # Apply InceptionResNetV2’s own preprocessing
    return inception_resnet_v2.preprocess_input(img299.astype(np.float32))

# 2) BUILD THE DATA ARRAYS
label_map = {'benign': 0, 'malignant': 1, 'normal': 2}
X_ir2, y_ir2 = [], []
for label, pairs in data.items():
    for img, _ in pairs:
        X_ir2.append(preprocess_ir2(img))
        y_ir2.append(label_map[label])
X_ir2 = np.array(X_ir2)
y_ir2 = np.array(y_ir2)

# 3) SPLIT INTO TRAIN / VAL / TEST
X_trval, X_test, y_trval, y_test = train_test_split(
    X_ir2, y_ir2, test_size=0.15, stratify=y_ir2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=42
)

# 4) COMPUTE CLASS WEIGHTS
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))

# 5) DATA AUGMENTATION & GENERATORS
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen  = ImageDataGenerator()
test_datagen = ImageDataGenerator()

batch_size = 32
train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
val_gen   = val_datagen.flow(  X_val,   y_val,   batch_size=batch_size, shuffle=False)
test_gen  = test_datagen.flow( X_test,  y_test,  batch_size=batch_size, shuffle=False)

# 6) BUILD THE InceptionResNetV2 BASE
# Try loading from internet; if not possible, fall back to your local file
local_weights = "/kaggle/input/inceptionresnetv2/keras/default/1/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
try:
    base = InceptionResNetV2(weights=local_weights, include_top=False, input_shape=(299,299,3))
except:
    base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))

# Freeze all base layers
for layer in base.layers:
    layer.trainable = False

# 7) ADD A NEW CLASSIFICATION HEAD
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=base.input, outputs=output)

# 8) COMPILE
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 9) SET UP CALLBACKS
es  = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# 10) TRAIN JUST THE HEAD
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weight,
    callbacks=[es, rlp]
)

# 11) FINE-TUNE ALL LAYERS
for layer in base.layers:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weight,
    callbacks=[es, rlp]
)

# 12) EVALUATE & REPORT
loss, acc = model.evaluate(test_gen)
print(f"InceptionResNetV2 Test Accuracy: {acc:.4f}")

preds = np.argmax(model.predict(test_gen), axis=1)
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("\nClassification Report:\n",
      classification_report(y_test, preds, target_names=["Benign","Malignant","Normal"]))

import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix

# 1) Evaluate on test set
eval_loss, eval_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_acc:.4f}")

# 2) Predictions and confusion matrix
preds = np.argmax(model.predict(test_gen), axis=1)
cm = confusion_matrix(y_test, preds)

# 3) Combine training histories (assuming you captured these)
train_acc   = history_head.history['accuracy']    + history_ft.history['accuracy']
val_acc     = history_head.history['val_accuracy']+ history_ft.history['val_accuracy']
train_loss  = history_head.history['loss']        + history_ft.history['loss']
val_loss    = history_head.history['val_loss']    + history_ft.history['val_loss']
epochs      = range(1, len(train_acc) + 1)

# 4) Plot confusion matrix & performance curves in a 1×2 faceted layout
fig, (ax_cm, ax_perf) = plt.subplots(1, 2, figsize=(16, 6))

# — Confusion Matrix —
im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
ax_cm.set_title('InceptionResNetV2 Confusion Matrix')
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha='center', va='center')
fig.colorbar(im, ax=ax_cm)

# — Training History —
ax_perf.plot(epochs, train_acc,   '-o', label='Train Acc')
ax_perf.plot(epochs, val_acc,     '--x', label='Val Acc')
ax_perf.plot(epochs, train_loss,  ':s', label='Train Loss')
ax_perf.plot(epochs, val_loss,    '-.d', label='Val Loss')
ax_perf.set_title('InceptionResNetV2 Training History')
ax_perf.set_xlabel('Epoch')
ax_perf.legend(loc='best')
ax_perf.grid(True)

plt.tight_layout()
plt.savefig('inceptionresnetv2_results.png', dpi=300)
plt.show()

# 5) Save evaluation metrics for later comparison
eval_metrics = {
    'InceptionResNetV2': {
        'loss': float(eval_loss),
        'accuracy': float(eval_acc)
    }
}
with open('inceptionresnetv2_evaluation.json', 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print("Saved plot to inceptionresnetv2_results.png")
print("Saved evaluation to inceptionresnetv2_evaluation.json")
