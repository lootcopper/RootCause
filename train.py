import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# Define paths
train_dir = '/Users/pranay/softwareDEV/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
validation_dir = '/Users/pranay/softwareDEV/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'

# Image preprocessing and augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # ResNet expects 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
validation_dataset = datasets.ImageFolder(validation_dir, transform=validation_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Define the model using ResNet50
class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CropDiseaseModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify output layer

        # Unfreeze some layers for fine-tuning
        for param in list(self.model.parameters())[:50]:
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

model = CropDiseaseModel(num_classes=len(train_dataset.classes))  

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Use Mixed Precision Training
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scaler = torch.amp.GradScaler() 
num_epochs = 35

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):  # FIXED: Use correct AMP mode
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'crop_disease_model.pth')
print("Model saved to crop_disease_model.pth")
