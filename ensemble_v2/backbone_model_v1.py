import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, save_path="model.pth"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Extract main logits
            outputs = model(images)
            logits = outputs.logits if isinstance(outputs, InceptionOutputs) else outputs

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load resampled data
data = np.load(r"resampled_data.npz")
X_resampled = torch.tensor(data['images'], dtype=torch.float32)
y_resampled = torch.tensor(data['labels'], dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_resampled, y_resampled)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load pretrained InceptionV3
inception_resnet_v2_model = models.inception_v3(pretrained=True)

# Replace the final fully connected layer for 5 classes
inception_resnet_v2_model.fc = nn.Linear(inception_resnet_v2_model.fc.in_features, 5)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.AdamW(inception_resnet_v2_model.parameters(), lr=0.001)

train_model(
    inception_resnet_v2_model,
    train_loader,
    criterion,
    optimizer,
    device='cuda',
    save_path="inception_resnet_v2_backbone.pth",
)
