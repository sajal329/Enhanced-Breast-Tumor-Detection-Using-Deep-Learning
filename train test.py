from sklearn.model_selection import train_test_split

# Suppose you have a list of (image, mask) pairs for each class
benign_data = data['benign']  # List of (image, mask) tuples
malignant_data = data['malignant']
normal_data = data['normal']

# Combine them and create labels
X = benign_data + malignant_data + normal_data  # All image/mask pairs
y = ([0]*len(benign_data)) + ([1]*len(malignant_data)) + ([2]*len(normal_data)) 
# 0 = benign, 1 = malignant, 2 = normal (or whichever labeling scheme you prefer)

# First split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Then split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)

print(f"Train size: {len(X_train)}")
print(f"Val size:   {len(X_val)}")
print(f"Test size:  {len(X_test)}")

##