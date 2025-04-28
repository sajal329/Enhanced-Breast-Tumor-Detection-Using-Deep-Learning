# 🎯 Enhanced Breast Tumor Detection Using Deep Learning

## 📖 Introduction
This project focuses on enhancing the detection of breast tumors using advanced deep learning techniques. It leverages pre-trained convolutional neural network (CNN) models like DenseNet, InceptionV3, InceptionResNetV2, and VGG architectures to achieve high accuracy in tumor classification tasks.

## 🧩 Table of Contents
- [📖 Introduction](#-introduction)
- [🧩 Table of Contents](#-table-of-contents)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [✨ Features](#-features)
- [📦 Dependencies](#-dependencies)
- [🛠️ Configuration](#-configuration)
- [📚 Documentation](#-documentation)
- [🔍 Examples](#-examples)
- [🛡️ Troubleshooting](#-troubleshooting)
- [👥 Contributors](#-contributors)
- [📝 License](#-license)

## ⚙️ Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/sajal329/Enhanced-Breast-Tumor-Detection-Using-Deep-Learning.git
    cd Enhanced-Breast-Tumor-Detection-Using-Deep-Learning
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
   *(Note: If `requirements.txt` is not present, see [Dependencies](#-dependencies) for manual installation.)*

## 🚀 Usage
- 📂 Prepare your dataset using `data read.py` and `Data_Prep.ipynb`.
- 🛠️ Choose and run the model training scripts like:
  - `VGG_16_19.ipynb`
  - `Comparing_Models.py`
  - `train test.py`
- 📊 Evaluate and compare model performances using the provided notebooks and scripts.

## ✨ Features
- 🔥 Utilizes powerful CNN architectures for breast tumor classification.
- 🖼️ Advanced pre-processing and augmentation techniques.
- 🧠 Comparative analysis of different deep learning models.
- 🛠️ Modular scripts and Jupyter notebooks for easy experimentation.

## 📦 Dependencies
The project requires:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install manually if needed:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

## 🛠️ Configuration
- Update dataset paths as required in the scripts.
- Hyperparameters such as batch size and learning rate can be modified within the model training scripts.

## 📚 Documentation
The repository includes:
- **Model Scripts**: Definitions for DenseNet, InceptionV3, InceptionResNetV2, VGG16, and VGG19.
- **Training Scripts**: `train test.py`, `VGG_16_19_with_deepnet.ipynb`.
- **Utilities**: Helper functions (`some func.py`) and data preparation scripts (`Data_Prep.ipynb`).

📘 For detailed workflow, refer to the Jupyter notebooks.

## 🔍 Examples
Train a VGG16 model:
```bash
python VGG16.py
```
Compare multiple models:
```bash
python Comparing_Models.py
```

## 🛡️ Troubleshooting
- ⚠️ **TensorFlow/Keras Version Issues**: Ensure your installed version matches hardware capabilities.
- 📂 **Dataset Errors**: Check the dataset structure and adjust paths accordingly.
- 🧹 **Memory Errors**: Lower the batch size or resize input images if necessary.

## 👥 Contributors
- [Sajal329](https://github.com/sajal329) - Main Author 🎉

## 📝 License
This project is licensed under the [MIT License](LICENSE).
