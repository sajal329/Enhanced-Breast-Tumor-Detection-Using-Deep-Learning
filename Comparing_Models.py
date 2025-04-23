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

# 2) Pairwise McNemar’s test
pvals = pd.DataFrame(index=models, columns=models, dtype=float)
bs_counts = pd.DataFrame(index=models, columns=models, dtype=int)  # b
cs_counts = pd.DataFrame(index=models, columns=models, dtype=int)  # c

for A, B in itertools.permutations(models, 2):
    corrA = (y_preds[A] == y_true)
    corrB = (y_preds[B] == y_true)
    b = np.sum(corrA & ~corrB)
    c = np.sum(~corrA & corrB)
    table = [[0, b],
             [c, 0]]
    res = mcnemar(table, exact=True)
    pvals.loc[A, B] = res.pvalue
    bs_counts.loc[A, B] = b
    cs_counts.loc[A, B] = c

# 3) Visualize the p-value matrix
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(pvals.astype(float), cmap='Reds', vmin=0, vmax=1)
# Annotate
for i, A in enumerate(models):
    for j, B in enumerate(models):
        if i!=j:
            txt = f"{pvals.loc[A,B]:.3f}"
            ax.text(j, i, txt, ha='center', va='center', color='black')
# Labels
ax.set_xticks(np.arange(len(models))); ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_yticks(np.arange(len(models))); ax.set_yticklabels(models)
ax.set_title("McNemar’s Test p-values (A vs B)")
fig.colorbar(im, ax=ax, label='p-value')
plt.tight_layout()
plt.savefig('mcnemar_pvalues_heatmap.png', dpi=300)
plt.show()

