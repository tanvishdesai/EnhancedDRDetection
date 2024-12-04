import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define custom dataset class for training dataset (.npz)
class CustomDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = torch.tensor(data['images'], dtype=torch.float32)
        self.y = torch.tensor(data['labels'], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define custom dataset class for test dataset (images folder)
class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = os.listdir(image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]  # return image and filename for saving predictions

# Define attention mechanism
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

# Define feature pyramid network (FPN)
class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        self.backbone = backbone
        self.lateral_layers = nn.ModuleList([nn.Conv2d(in_channels, 256, kernel_size=1) for in_channels in [256, 512, 1024, 2048]])
        self.smooth_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in range(4)])

    def forward(self, x):
        features = self.backbone(x)
        pyramidal_features = []
        for i, feature in enumerate(features):
            lateral = self.lateral_layers[i](feature)
            pyramidal_features.append(lateral)
        merged_features = sum(pyramidal_features)
        return self.smooth_layers[0](merged_features)

# Define Bayesian dropout layer
class BayesianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(BayesianDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

# Define Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        target = output[:, class_idx]
        target.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1)
        cam = nn.functional.relu(cam)
        return cam

# Define custom model with FPN, Attention, Bayesian Dropout
class CustomModel(nn.Module):
    def __init__(self, backbone, num_classes=5):
        super(CustomModel, self).__init__()
        self.backbone = backbone
        self.fpn = FPN(self.backbone)
        self.attention = Attention(in_channels=256)
        self.bayesian_dropout = BayesianDropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fpn(x)
        x = self.attention(x)
        x = self.bayesian_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define model ensemble class
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

# Load pre-trained models (already trained and saved)
resnet_model = torch.load('resnet18_model.pth').to(device)
vgg_model = torch.load('vgg16_model.pth').to(device)
swin_model = torch.load('swin_transformer_model.pth').to(device)

# Create ensemble model
ensemble_model = EnsembleModel([resnet_model, vgg_model, swin_model]).to(device)

# Define evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate(model, dataloader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')

    return accuracy, f1, precision, recall

# Load training dataset (.npz)
train_dataset = CustomDataset('train_data.npz')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load test dataset (images folder)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = TestDataset('test_images', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define optimizer and loss
optimizer = optim.AdamW(ensemble_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    ensemble_model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = ensemble_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}')

# Save model checkpoint
torch.save(ensemble_model.state_dict(), 'ensemble_model.pth')

# Evaluate the model
accuracy, f1, precision, recall = evaluate(ensemble_model, test_loader)
print(f'Evaluation - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}')

# Save predictions for test images
ensemble_model.eval()
test_preds = []
test_filenames = []
with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = ensemble_model(images)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_filenames.extend(filenames)

# Save predictions to CSV file
submission_df = pd.DataFrame({'filename': test_filenames, 'prediction': test_preds})
submission_df.to_csv('test_predictions.csv', index=False)

print('Predictions saved to test_predictions.csv')

# Grad-CAM Example (visualization for one image)
grad_cam = GradCAM(model=ensemble_model, target_layer=ensemble_model.backbone.layer4)

# Visualizing Grad-CAM for a test image (choose an image index)
sample_image, _ = test_dataset[0]
sample_image = sample_image.unsqueeze(0).to(device)
output = ensemble_model(sample_image)
class_idx = torch.argmax(output, dim=1).item()
heatmap = grad_cam.generate_heatmap(sample_image, class_idx)

# Visualizing the heatmap on the image
plt.imshow(heatmap.squeeze().cpu(), cmap='jet', alpha=0.5)
plt.imshow(sample_image.squeeze().cpu().permute(1, 2, 0).numpy(), alpha=0.5)
plt.show()