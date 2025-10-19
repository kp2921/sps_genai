import torch
from torchvision import transforms
from PIL import Image
from fastapi import UploadFile, File
from helper_lib.model import load_model

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained CNN model
model = load_model("checkpoint/best/model_epoch_005.pth", device)

# Define CIFAR-10 classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

async def predict_cnn(file: UploadFile = File(...)):
    """Classify an uploaded image using the trained CNN."""
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = CLASSES[predicted.item()]
    return {"filename": file.filename, "prediction": predicted_class}
