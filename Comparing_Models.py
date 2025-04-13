import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Map model names to JSON filenames
model_files = {
    'VGG16':           'results_vgg16.json',
    'VGG19':           'results_vgg19.json',
    'InceptionV3':     'results_inceptionV3.json',
    'DenseNet169':     'results_densenet169.json',
    'InceptionResNetV2':'results.json'
}

# 2) Load all results
results = {}
for name, fname in model_files.items():
    with open(fname, 'r') as f:
        results[name] = json.load(f);

# 3) Build DataFrames
# Overall accuracy
accuracy = pd.Series({m: r['classification_report']['accuracy']
                      for m, r in results.items()})

# Macro-avg metrics
macro = pd.DataFrame({
    m: r['classification_report']['macro avg']
    for m, r in results.items()
}).T

# Weighted-avg metrics
weighted = pd.DataFrame({
    m: r['classification_report']['weighted avg']
    for m, r in results.items()
}).T

# Per-class F1
classes = ['benign', 'malignant', 'normal']
f1_per_class = pd.DataFrame({
    m: [results[m]['classification_report'][cls]['f1-score']
         for cls in classes]
    for m in results
}, index=classes).T

models = list(results.keys())

# === 1) Line plot: Accuracy vs Macro metrics ===
plt.figure(figsize=(12, 5))
plt.plot(models, accuracy,     '-o', label='Accuracy')
plt.plot(models, macro['precision'], '--x', label='Macro Precision')
plt.plot(models, macro['recall'],    '--x', label='Macro Recall')
plt.plot(models, macro['f1-score'],  '--x', label='Macro F1-score')
plt.title('Overall Accuracy & Macro-averaged Metrics')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_vs_macro.png')
plt.show()

# === 2) Radar chart: Macro Precision/Recall/F1 ===
labels = ['precision', 'recall', 'f1-score']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
for m in models:
    vals = macro.loc[m, labels].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, '-o', label=m)
ax.set_thetagrids(np.degrees(angles), labels + [labels[0]])
ax.set_title('Macro Metrics Radar Chart', y=1.1)
ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0))
plt.tight_layout()
plt.savefig('macro_radar.png')
plt.show()

# === 3) Grouped bar chart: Weighted-average metrics ===
ind = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(ind,               weighted['precision'], width, label='Precision')
ax.bar(ind + width,       weighted['recall'],    width, label='Recall')
ax.bar(ind + 2*width,     weighted['f1-score'],  width, label='F1-score')

ax.set_xticks(ind + width)
ax.set_xticklabels(models)
ax.set_title('Weighted-average Metrics by Model')
ax.set_ylabel('Score')
ax.legend()
plt.tight_layout()
plt.savefig('weighted_metrics_grouped.png')
plt.show()

# === 4) Grouped bar chart: Per-class F1-scores ===
ind = np.arange(len(classes))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))
for i, m in enumerate(models):
    ax.bar(ind + i*width,
           f1_per_class.loc[m],
           width,
           label=m)

ax.set_xticks(ind + width*(len(models)-1)/2)
ax.set_xticklabels(classes)
ax.set_title('Per-Class F1-Score Comparison')
ax.set_ylabel('F1-score')
ax.legend()
plt.tight_layout()
plt.savefig('per_class_f1_grouped.png')
plt.show()
