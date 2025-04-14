import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Map model names to JSON filenames
model_files = {
    'VGG16':            'results_vgg16.json',
    'VGG19':            'results_vgg19.json',
    'InceptionV3':      'results_inceptionV3.json',
    'DenseNet169':      'results_densenet169.json',
    'InceptionResNetV2':'results.json'
}

# 2) Load results
results = {m: json.load(open(f)) for m, f in model_files.items()}

# 3) Build DataFrames
accuracy = pd.Series({m: r['classification_report']['accuracy'] 
                      for m, r in results.items()})
weighted = pd.DataFrame({m: r['classification_report']['weighted avg'] 
                         for m, r in results.items()}).T
macro    = pd.DataFrame({m: r['classification_report']['macro avg'] 
                         for m, r in results.items()}).T
classes = ['benign','malignant','normal']
f1_pc = pd.DataFrame({
    m: [results[m]['classification_report'][cls]['f1-score'] for cls in classes]
    for m in results
}, index=classes).T
models = list(results.keys())

# === 1) Line plot: Accuracy vs Weighted F1 ===
plt.figure(figsize=(12,5))
x = np.arange(len(models))
acc_vals = accuracy.values
wf1_vals = weighted['f1-score'].values

plt.plot(x, acc_vals, '-o', label='Accuracy', linewidth=2)
plt.fill_between(x, acc_vals-0.01, acc_vals+0.01, alpha=0.1)
plt.plot(x, wf1_vals, '--x', label='Weighted F1-Score', linewidth=2)
plt.fill_between(x, wf1_vals-0.01, wf1_vals+0.01, alpha=0.1)

# annotate peaks
for xi, y in zip(x, acc_vals):
    plt.text(xi, y+0.005, f"{y:.2f}", ha='center')
for xi, y in zip(x, wf1_vals):
    plt.text(xi, y-0.015, f"{y:.2f}", ha='center')

plt.xticks(x, models)
plt.title('Accuracy vs. Weighted F1-Score', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.ylim(0,1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_vs_weightedF1.png', dpi=300)
plt.savefig('accuracy_vs_weightedF1.svg')
plt.show()

# === 2) Radar chart: Macro Precision/Recall/F1 ===
labels = ['precision','recall','f1-score']
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
for m in models:
    vals = macro.loc[m, labels].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, '-o', linewidth=1.5, label=m)
    ax.fill(angles, vals, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks([0.5,0.75,1.0])
ax.set_title('Macro-averaged Metrics Radar', y=1.1, fontsize=14)
ax.legend(bbox_to_anchor=(1.3,0.0), fontsize=8)
plt.tight_layout()
plt.savefig('macro_radar.png', dpi=300)
plt.savefig('macro_radar.svg')
plt.show()

# === 3) Grouped bar chart: Weighted-average metrics ===
ind = np.arange(len(models))
w = 0.25

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(ind,               weighted['precision'], w, label='Precision')
ax.bar(ind + w,           weighted['recall'],    w, label='Recall')
ax.bar(ind + 2*w,         weighted['f1-score'],  w, label='F1-Score')

ax.set_xticks(ind + w)
ax.set_xticklabels(models)
ax.set_title('Weighted-Average Metrics by Model', fontsize=14)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0,1)
ax.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('weighted_metrics_grouped.png', dpi=300)
plt.savefig('weighted_metrics_grouped.svg')
plt.show()

# === 4) Grouped bar chart: Per-class F1-scores ===
ind = np.arange(len(classes))
w = 0.15

fig, ax = plt.subplots(figsize=(10,6))
for i, m in enumerate(models):
    ax.bar(ind + i*w, f1_pc.loc[m], w, label=m)

ax.set_xticks(ind + w*(len(models)-1)/2)
ax.set_xticklabels(classes)
ax.set_title('Per-Class F1-Score Comparison', fontsize=14)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_ylim(0,1)
ax.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('per_class_f1_grouped.png', dpi=300)
plt.savefig('per_class_f1_grouped.svg')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# === Concentric radial bar chart: Per-class F1 Scores ===
# classes & models should already be defined:
#   classes = ['benign', 'malignant', 'normal']
#   models = ['VGG16', 'VGG19', 'InceptionV3', 'DenseNet169', 'InceptionResNetV2']
#   f1_per_class: DataFrame indexed by model, columns=classes

num_models  = len(models)
num_classes = len(classes)

# Radii for each class ring
radii = np.arange(1, num_classes + 1) * 1.5
bar_width = 2 * np.pi / num_models * 0.8
angles    = np.linspace(0, 2*np.pi, num_models, endpoint=False)

# Pick a qualitative colormap for distinct model colors
colors = plt.cm.tab10(np.linspace(0, 1, num_models))

fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, polar=True)

for i, cls in enumerate(classes):
    r = radii[i]
    vals = [f1_per_class.loc[m, cls] for m in models]
    for j, (angle, val) in enumerate(zip(angles, vals)):
        ax.bar(angle, val,
               width=bar_width,
               bottom=r,
               color=colors[j],
               edgecolor='white',
               linewidth=0.8)
        # Annotate each bar tip
        ax.text(angle, r + val + 0.05, f"{val:.2f}",
                ha='center', va='bottom', fontsize=9)

# Tidy up
ax.set_yticks(radii)
ax.set_yticklabels(classes, fontsize=11)   # ring labels = classes
ax.set_xticks([])                          # hide angle ticks
ax.set_ylim(0, radii[-1] + 1)
ax.grid(False)
ax.spines['polar'].set_visible(False)

# Legend for models
handles = [plt.Line2D([0], [0], color=colors[i], lw=4) 
           for i in range(num_models)]
ax.legend(handles, models,
          loc='upper right',
          bbox_to_anchor=(1.3, 1.05),
          fontsize=9)

ax.set_title('Per-Class F1 Scores Across Models', y=1.08, fontsize=14)

plt.tight_layout()
plt.savefig('concentric_radial_f1.png', dpi=300)
plt.savefig('concentric_radial_f1.svg')
plt.show()
