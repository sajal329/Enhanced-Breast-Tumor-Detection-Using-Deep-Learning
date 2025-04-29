# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rotation_range=20,        # rotate images by up to 20 degrees
    width_shift_range=0.1,    # shift horizontally by 10%
    height_shift_range=0.1,   # shift vertically by 10%
    zoom_range=0.2,           # zoom in/out by 20%
    horizontal_flip=True,     # flip horizontally
    fill_mode='nearest'       # fill in missing pixels#
)

# For validation and test, we usually do NOT augment, just the same normalization
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True
)

val_generator = val_datagen.flow(
    X_val, y_val,
    batch_size=batch_size,
    shuffle=False
)

test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=batch_size,
    shuffle=False
)
