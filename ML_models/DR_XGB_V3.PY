# imports
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import optuna

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])  # First column has image names
        label = int(self.data.iloc[idx, 1])  # Second column has labels

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Feature Extractor Model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        self.feature_layers = nn.Sequential(*list(resnet.children())[:-1])  # Exclude final classification layer

    def forward(self, x):
        x = self.feature_layers(x)
        return torch.flatten(x, 1)


# Progress Bar
def extract_features_with_progress(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, target_labels in tqdm(dataloader, desc="Extracting Features", unit="batch"):
            images = images.to(device)
            target_labels = target_labels.to(device)

            outputs = model(images)
            features.append(outputs.cpu())
            labels.append(target_labels.cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


# Hyperparameter tuning
def train_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    trial.set_user_attr("model", model)
    y_proba = model.predict_proba(X_val)

    # Class and sample matrix for ROC-AUC
    y_true_binarized = label_binarize(y_val, classes=list(range(len(set(y_val)))))

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y_true_binarized, y_proba, multi_class='ovr')

    return roc_auc


# Evaluation
def evaluate_model(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    if y_proba is not None:  # Compute ROC-AUC only if probabilities are available
        y_true_binarized = label_binarize(y_true, classes=list(range(len(set(y_true)))))
        roc_auc = roc_auc_score(y_true_binarized, y_proba, multi_class='ovr')
        print("ROC-AUC score: ", roc_auc)
    
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)


# Main function
if __name__ == "__main__":
    batch_size = 16
    image_size = 224


# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Paths
data_path = "diabetic retinopathy/train/train"  # Path to the folder with images
csv_path = "ML_models/trainLabels_ensemble.csv"  # Path to the CSV file with labels

# Dataloader
dataset = CustomDataset(csv_file=csv_path, image_folder=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialization
feature_extractor = FeatureExtractor().to(device)
features, labels = extract_features_with_progress(feature_extractor, dataloader, device)

features = features.numpy()
labels = labels.numpy()

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)


# Training
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: train_xgboost(trial, X_train, y_train, X_val, y_val), n_trials=10)

# Evaluate the best model
best_model = study.best_trial.user_attrs["model"]
y_pred = best_model.predict(X_val)
evaluate_model(y_val, y_pred)

best_model.save_model("best_model.json")