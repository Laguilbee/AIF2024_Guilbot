import torch
import gradio as gr
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torchvision.models as models
import requests
import json
import io
from gradio.components import Image

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def init_model(device):
    mobilenet = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model = nn.Sequential(mobilenet.features,mobilenet.avgpool,nn.Flatten())
    model = model.to(device)
    model.eval()
    return model

model = init_model(device)

def normalize_image(image):
    # Convertir l'image NumPy en image PIL si nécessaire
    image = Image.fromarray(image)
    # Appliquer la transformation
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def extract_features(image, model):
    image_tensor = normalize_image(image)
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy()

def predict(image):
    feature_vector = extract_features(image, model).flatten()
    response = requests.post(
        "http://127.0.0.1:5000/recommend", 
        json={"vector": feature_vector.tolist()}
    )

    if response.status_code == 200:
        list_path = response.json()
        images = []
        for path in list_path:
            image = Image.open(path)
            images.append(image)

        return images 
    else:
        print(f"Erreur API: {response.status_code}")
        return f"Erreur API: {response.status_code}"


if __name__ == '__main__':
    
    gr.Interface(
        fn=predict,
        inputs="image",
        outputs=[Image(type="pil") for i in range(5)],
        live=True,
        description="Upload an image to get recommendations."
    ).launch(debug=True, share=True,server_port=8000)

