import numpy as np
import cv2
from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics           import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications        import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras import layers, models, optimizers, callbacks

# 1) PREPROCESS FUNCTION FOR DenseNet169 (224×224 + 3-channel + correct normalization)
def preprocess_densenet(img):
    img224 = cv2.resize(img, (224, 224))
    if img224.ndim == 2:  # grayscale → 3 channels
        img224 = np.stack([img224]*3, axis=-1)
    return densenet_preprocess(img224.astype(np.float32))

# 2) BUILD X_dn, y_dn ARRAYS
label_map = {'benign': 0, 'malignant': 1, 'normal': 2}
X_dn, y_dn = [], []
for label, pairs in data.items():
    for img, _ in pairs:
        X_dn.append(preprocess_densenet(img))
        y_dn.append(label_map[label])
X_dn = np.array(X_dn)
y_dn = np.array(y_dn)

# 3) SPLIT INTO TRAIN / VAL / TEST
X_trval, X_test, y_trval, y_test = train_test_split(
    X_dn, y_dn, test_size=0.15, stratify=y_dn, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.15, stratify=y_trval, random_state=42
)

# 4) COMPUTE CLASS WEIGHTS TO MITIGATE IMBALANCE
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))

# 5) DATA AUGMENTATION GENERATORS
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
# 6) BUILD DenseNet169 BASE (WITH LOCAL WEIGHTS IF NEEDED)
local_weights_path = "/kaggle/input/densenet169/tensorflow2/default/1/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5"
try:
    base = DenseNet169(weights=local_weights_path, include_top=False, input_shape=(224,224,3))
except:
    base = DenseNet169(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze all base layers initially
for layer in base.layers:
    layer.trainable = False

# 7) ADD CLASSIFICATION HEAD
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=base.input, outputs=outputs)

# 8) COMPILE
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 9) SET UP CALLBACKS
es  = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# 10) TRAIN HEAD ONLY
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

# 12) EVALUATE ON TEST SET
loss, acc = model.evaluate(test_gen)
print(f"DenseNet169 Test Accuracy: {acc:.4f}")

# 13) CONFUSION MATRIX & CLASSIFICATION REPORT
preds = np.argmax(model.predict(test_gen), axis=1)
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Benign","Malignant","Normal"]))


import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix

# 1) Re-evaluate to get scalar test metrics in fresh variables
eval_loss, eval_acc = model.evaluate(test_gen, verbose=0)

# 2) Get predictions and build confusion matrix
preds = np.argmax(model.predict(test_gen), axis=1)
cm = confusion_matrix(y_test, preds)

# 3) Combine your two training histories
train_acc   = history_head.history['accuracy']    + history_ft.history['accuracy']
val_acc     = history_head.history['val_accuracy']+ history_ft.history['val_accuracy']
train_loss  = history_head.history['loss']        + history_ft.history['loss']
val_loss    = history_head.history['val_loss']    + history_ft.history['val_loss']
epochs      = range(1, len(train_acc) + 1)

# 4) Plot confusion matrix & performance side by side
fig, (ax_cm, ax_perf) = plt.subplots(1, 2, figsize=(14, 6))

# — Confusion Matrix —
im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
ax_cm.set_title('DenseNet169 Confusion Matrix')
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha='center', va='center')
fig.colorbar(im, ax=ax_cm)

# — Accuracy & Loss Curves —
ax_perf.plot(epochs,      train_acc,   '-o', label='Train Acc')
ax_perf.plot(epochs,      val_acc,     '--x', label='Val Acc')
ax_perf.plot(epochs,      train_loss,  ':s', label='Train Loss')
ax_perf.plot(epochs,      val_loss,    '-.d', label='Val Loss')
ax_perf.set_title('DenseNet169 Training History')
ax_perf.set_xlabel('Epoch')
ax_perf.legend(loc='best')
ax_perf.grid(True)

plt.tight_layout()
plt.savefig('densenet169_results.png', dpi=300)
plt.show()

# 5) Save test‐set metrics for later comparison
eval_metrics = {
    'DenseNet169': {
        'loss': float(eval_loss),
        'accuracy': float(eval_acc)
    }
}
with open('densenet169_evaluation.json', 'w') as f:
    json.dump(eval_metrics, f, indent=2)

print("Saved plot to densenet169_results.png")
print("Saved evaluation to densenet169_evaluation.json")

