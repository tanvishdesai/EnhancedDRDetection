"""this is supposed to be my primary model which follows ensemble learning : combining VGGNET, RESNET & swine transformer
it is very computaionaly heavy each spoch has 1842 batches and 1 hour teaches about 550 batches,
so maybe around 4 hours of training for 1 epoch, min 20 epochs required for considerable results,
so maybe just add the explaination of this model along the theoratical accuracy, training this model may not be possible"""

# imports
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet18
from timm import create_model
from torch.optim.lr_scheduler import StepLR

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)  
        self.image_folder = image_folder  
        self.transform = transform        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        
        image = Image.open(img_name).convert("RGB")

        
        if self.transform:
            image = self.transform(image)

        return image, label

# Architecture of the model
class EnhancedDRDetectionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedDRDetectionModel, self).__init__()

        # VGG
        self.vgg_features = nn.Sequential(*list(vgg16(pretrained=True).features)[:10])

        # ResNet
        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Swin Transformer
        self.swin_transformer = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)

        # Final classification layer
        self.fc = nn.Linear(401408 + 512 + 1024, num_classes)  # Adjusted based on feature sizes

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # VGG features
        vgg_out = self.vgg_features(x)
        vgg_out = torch.flatten(vgg_out, start_dim=1)

        # ResNet features
        resnet_out = self.resnet(x)
        resnet_out = torch.flatten(resnet_out, start_dim=1)

        # Swin Transformer featues
        swin_out = self.swin_transformer(x)

        # Concatenate features
        combined_features = torch.cat((vgg_out, resnet_out, swin_out), dim=1)

        # # Debugging shapes
        # print("VGG output shape:", vgg_out.shape)
        # print("ResNet output shape:", resnet_out.shape)
        # print("Swin Transformer output shape:", swin_out.shape)
        # print("Combined features shape:", combined_features.shape)

        #  Dropout and classification layer
        combined_features = self.dropout(combined_features)
        out = self.fc(combined_features)

        return out

# Paths
csv_file = 'ML_models/trainLabels_ensemble.csv'
image_folder = r'.\diabetic retinopathy\train\train'

# Data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# dataloader
dataset = CustomDataset(csv_file=csv_file, image_folder=image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = EnhancedDRDetectionModel(num_classes=5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization via weight_decay
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs


# Training loop
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("Device:", device)
print("Model initialized and moved to device.")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} started.") 
    model.train()
    epoch_loss = 0
    for i, (images, labels) in enumerate(dataloader):
        if i % 10 == 0:
            print(f"Batch {i} done.")   #having 1842 batches per epoch, we decided to take print every 10 batches so that we can confirm
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")