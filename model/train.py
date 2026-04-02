import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import os

# ==============================
# GPU Setup
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ==============================
# Dataset Paths
# ============================== 

train_dir = "../dataset_split/train"
val_dir = "../dataset_split/validation"
test_dir = "../dataset_split/test"

# ==============================
# Image Transform
# ==============================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ==============================
# Load Dataset
# ==============================

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)

print(train_dataset.class_to_idx)
# ==============================
# DataLoader
# ==============================

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==============================
# Load Model
# ==============================

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Train only last layer
for param in model.fc.parameters():
    param.requires_grad = True

# ==============================
# Loss & Optimizer
# ==============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# ==============================
# Training
# ==============================

epochs = 5
best_accuracy = 0

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-"*30)

    # Training
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress
        progress = (i / len(train_loader)) * 100
        print(f"Training Progress: {progress:.2f}%", end="\r")

    train_accuracy = 100 * correct / total

    print(f"\nTraining Loss: {running_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    # ==============================
    # Validation
    # ==============================

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save Best Model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "deepfake_model.pth")
        print("Best Model Saved!")

print("\nTraining Complete!")
print("Best Validation Accuracy:", best_accuracy)