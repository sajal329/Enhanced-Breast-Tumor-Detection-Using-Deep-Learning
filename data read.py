import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Adjust this path to match your own Kaggle environment/dataset path.
# For example, if the dataset is in ../input/breast-ultrasound-images-dataset/
BASE_PATH = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"

# We'll keep our data in a dictionary: { 'benign': [], 'malignant': [], 'normal': [] }
# Each entry will be a list of tuples: (image_array, mask_array).
data = {
    'benign': [],
    'malignant': [],
    'normal': []
}

# Helper function to load images and masks
def load_images_and_masks(folder_path):
    """
    Given a folder path, returns a list of (image, mask) tuples.
    It assumes every image has a corresponding mask file with '_mask' appended before the extension.
    """
    # Gather all PNG files that are NOT masks
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.endswith(".png") and "_mask" not in f]

    loaded_data = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        # Construct mask filename: 
        # e.g., if 'X.png' is the image, we expect 'X_mask.png' for the mask
        mask_file = img_file.replace(".png", "_mask.png")
        mask_path = os.path.join(folder_path, mask_file)

        # Read the image and mask (grayscale or color depending on your need)
        # Ultrasound is often grayscale, so let's do IMREAD_GRAYSCALE
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is not None and mask is not None:
            loaded_data.append((image, mask))
        else:
            print(f"Warning: Could not read image/mask for {img_file}")

    return loaded_data

# Loop through each subfolder and load data
for label in data.keys():
    folder = os.path.join(BASE_PATH, label)
    data[label] = load_images_and_masks(folder)

# Now let's print a summary
print("Dataset Summary")
print("---------------")
total_images = 0
for label in data.keys():
    count = len(data[label])
    total_images += count
    # Print basic shape info for the first sample (just to give an idea)
    if count > 0:
        sample_img, sample_mask = data[label][0]
        print(f"Class: {label}")
        print(f"  Number of samples: {count}")
        print(f"  Sample image shape: {sample_img.shape}")
        print(f"  Sample mask shape:  {sample_mask.shape}")
    else:
        print(f"Class: {label} - No images found.")

print("---------------")
print(f"Total images loaded: {total_images}")

# Optional: Display a small example of an image + mask
# Let's pick one from the 'benign' category (if it exists)
if len(data['benign']) > 0:
    sample_img, sample_mask = data['benign'][0]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(sample_img, cmap='gray')
    plt.title("Benign Sample Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(sample_mask, cmap='gray')
    plt.title("Corresponding Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()