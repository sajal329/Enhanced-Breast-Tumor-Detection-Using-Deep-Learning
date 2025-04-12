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
