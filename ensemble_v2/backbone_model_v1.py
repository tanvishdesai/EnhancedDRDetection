import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from timm import create_model
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load resampled data
data = np.load(r"C:\Users\DELL\Downloads\resampled_data.npz")
X_resampled = torch.tensor(data['images'], dtype=torch.float32)
y_resampled = torch.tensor(data['labels'], dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_resampled, y_resampled)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=5)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Train Swin Transformer
train_model(model, train_loader, criterion, optimizer, device='cuda', save_path="swin_backbone.pth")