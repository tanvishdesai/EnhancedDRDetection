# imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential                                                                              # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam                                                                                # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator                                                         # type: ignore
from tensorflow.keras.utils import to_categorical, Sequence                                                                 # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping                                                     # type: ignore
from PIL import Image


# Paths
csv_file = 'ML_models/trainLabels_ensemble.csv'
image_folder = r'diabetic retinopathy/train/train'

# function for plotting
def plot_training_history(history):
    # Plot Accuracy
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load Dataset
data = pd.read_csv(csv_file)
data['level'] = data['level'].astype(int)  # Ensure levels are integers

# Custom Dataset
class CustomDataGenerator(Sequence):
    def __init__(self, dataframe, image_folder, batch_size, target_size=(224, 224), augment=False, **kwargs):
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment

        # Data Augmentations
        self.datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20 if augment else 0,
            width_shift_range=0.2 if augment else 0,
            height_shift_range=0.2 if augment else 0,
            shear_range=0.2 if augment else 0,
            zoom_range=0.2 if augment else 0,
            horizontal_flip=augment
        )

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, levels = [], []
        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.image_folder, row['image'])
            image = Image.open(img_path).convert("RGB").resize(self.target_size)
            image = np.array(image)
            images.append(image)
            levels.append(row['level'])
        images = np.array(images)
        levels = to_categorical(np.array(levels), num_classes=self.dataframe['level'].nunique())
        return self.datagen.flow(images, levels, batch_size=self.batch_size).__next__()

# Split
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

train_generator = CustomDataGenerator(train_data, image_folder, batch_size=32, augment=True)
val_generator = CustomDataGenerator(val_data, image_folder, batch_size=32)

# Keras Sequential Model
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data['level'].nunique(), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train Loop
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

# Save the Model
model.save('DiabeticRetinopathyCustomModel.h5')
print("Model saved as DiabeticRetinopathyCustomModel.h5.")

plot_training_history(history)

# Evaluation
y_true = val_data['level'].values
val_generator = CustomDataGenerator(val_data, image_folder, batch_size=32, augment=False)
y_pred = np.argmax(model.predict(val_generator, verbose=1), axis=1)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


print("\nClassification Report:")
print(classification_report(y_true, y_pred))