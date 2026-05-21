# Common Preprocessing and SMOTE
import os
import pandas as pd
import numpy as np
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

# Paths
csv_file = r"C:\Users\DELL\Downloads\train_1_modified.csv"
image_folder = r"C:\Users\DELL\Downloads\archive\train_images\train_images"

# Load dataset
data = pd.read_csv(csv_file)
X = data['id_code']
y = data['diagnosis']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Transform for all models
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract features from images
def extract_image_features(image_paths, transform, folder=image_folder):
    images = []
    for img_path in image_paths:
        # Add .png extension to the image path
        img_path = img_path
        img = Image.open(os.path.join(folder, img_path)).convert("RGB")
        img = transform(img)
        images.append(img.numpy())
    return np.array(images)


X_train_images = extract_image_features(X_train, transform)

# Flatten image data for SMOTE
X_train_flat = X_train_images.reshape(X_train_images.shape[0], -1)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back to image form
X_resampled = X_resampled.reshape(-1, 3, 299, 299)

# Save resampled data
np.savez('resampled_data.npz', images=X_resampled, labels=y_resampled)
