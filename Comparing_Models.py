import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Map model names to JSON filenames
model_files = {
    'VGG16':           '/kaggle/input/model-metrics/results_vgg16.json',
    'VGG19':           '/kaggle/input/model-metrics/results_vgg19.json',
    'InceptionV3':     '/kaggle/input/model-metrics/results_incceptionV3.json',
    'DenseNet169':     '/kaggle/input/model-metrics/results_densenet169.json',
    'InceptionResNetV2':'/kaggle/input/model-metrics/results.json'
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
plt.savefig('accuracy_vs_macro.svg')
plt.show()

# === 2) Weighted-average metrics: Grouped bar chart ===
labels = ['precision', 'recall', 'f1-score']
metrics = weighted[labels]

ind   = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))
for i, metric in enumerate(labels):
    ax.bar(ind + i*width,
           metrics[metric],
           width,
           label=metric.capitalize())

# X-axis ticks & labels
ax.set_xticks(ind + width)
ax.set_xticklabels(models, fontsize=11)

# Titles & limits
ax.set_title('Weighted-Average Metrics by Model', fontsize=14)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.0)

# Grid & legend
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('weighted_metrics_bar_clean.png', dpi=300)
plt.savefig('weighted_metrics_bar_clean.svg')
plt.show()

# === 3) Radar chart: Per-class F1-Score Comparison (Clean + Zoomed) ===
labels = ['benign', 'malignant', 'normal']
# compute the angles
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)

# Plot each model without any fill
for m in models:
    vals = f1_per_class.loc[m, labels].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, '-o', label=m, linewidth=1)

# Radial gridlines at useful F1 thresholds
ax.set_rgrids([0.5, 0.7, 0.9], angle=45)
# Zoom in to the relevant range
ax.set_ylim(0.45, 1.0)

# Clean up background & spine
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.patch.set_alpha(0)
ax.spines['polar'].set_visible(False)

# Angular labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)

# Title & legend
ax.set_title('Per-class F1-Score Radar', y=1.08, fontsize=14)
ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), fontsize=8)

plt.tight_layout()
plt.savefig('per_class_f1_radar_clean.png', dpi=300)
plt.savefig('per_class_f1_radar_clean.svg')
plt.show()

# === 4) Radar chart: Models as axes, Per-class Precision across models (Clean + Zoomed) ===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you already have:
#   models           = ['VGG16', 'VGG19', 'InceptionV3', 'DenseNet169', 'InceptionResNetV2']
#   results          = {...}  # your loaded JSON dict
#   classes          = ['benign', 'malignant', 'normal']

# 1) Build the per-class Precision DataFrame
precision_per_class = pd.DataFrame({
    m: [ results[m]['classification_report'][cls]['recall'] for cls in classes ]
    for m in models
}, index=classes).T

# 2) Compute angles for the radar
labels = models
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 3) Plot
fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, polar=True)

for cls in classes:
    vals = precision_per_class.loc[:, cls].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, '-o', label=cls.capitalize(), linewidth=1)
    # optional fill:
    # ax.fill(angles, vals, alpha=0.1)

# 4) Styling: zoom & clean
ax.set_rgrids([0.5, 0.7, 0.9], angle=45)
ax.set_ylim(0.45, 1.0)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.patch.set_alpha(0)
ax.spines['polar'].set_visible(False)

# 5) Labels, title, legend
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_title('Per-Class Recall Across Models', y=1.08, fontsize=14)
ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), fontsize=8)

plt.tight_layout()
plt.savefig('recall_per_class_across_models_radar.png', dpi=300)
plt.savefig('recall_per_class_across_models_radar.svg')
plt.show()
