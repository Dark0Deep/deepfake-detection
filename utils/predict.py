import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import cv2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# SAME as training
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(
    torch.load("model/deepfake_model.pth", map_location=device)
)

model = model.to(device)
model.eval()

classes = ["Fake", "Real"]


def predict(face):

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

    label = classes[predicted.item()]
    confidence = confidence.item()

    return label, confidence