import torch
import gradio as gr
import torch.nn as nn
from PIL import Image as PILImage
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torchvision.models as models
import requests
from gradio.components import Image as GradioImage
from transformers import DistilBertModel,DistilBertTokenizerFast
import re

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

def init_distilbert_model(device):
    tokenizer_model='distilbert-base-uncased'
    model = DistilBertModel.from_pretrained(tokenizer_model)
    model.to(device)
    model.eval()
    return model

model = init_model(device)
distilBertModel = init_distilbert_model(device)


def normalize_image(image):
    # Convertir l'image NumPy en image PIL si nécessaire
    image = PILImage.fromarray(image)
    # Appliquer la transformation
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def extract_features(image, model):
    image_tensor = normalize_image(image)
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy()

def clean_text(text):
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir en minuscules
    text = text.lower()
    return text

def predict_images_with_image(image):
    feature_vector = extract_features(image, model).flatten()
    #print("feature_vector = ",feature_vector)
    response = requests.post(
        "http://annoy-db:5066/recommend", 
        json={"vector": feature_vector.tolist(),"image_bool":True}
    )

    if response.status_code == 200:
        list_path = response.json()
        images = []
        for path in list_path:
            image = PILImage.open(path)
            images.append(image)

        return images 
    else:
        print(f"Erreur API: {response.status_code}")
        return f"Erreur API: {response.status_code}"

def predict_images_with_text(description):
    description = clean_text(description)
    tokenizer_model='distilbert-base-uncased'
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_model)

    tokenized_input = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = distilBertModel(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        embeddings = last_hidden_states.cpu().numpy().flatten()


    response = requests.post(
        "http://annoy-db:5066/recommend", 
        json={"vector": embeddings.tolist(),"image_bool":False}
    )

    print("reponse : ",response)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur API: {response.status_code}")
        return f"Erreur API: {response.status_code}"


if __name__ == '__main__':


    with gr.Blocks() as demo:
        gr.Markdown("Système de recommandation.")
        with gr.Tab("Recommandation de film par synopsis"):
            text_input = gr.Textbox()
            text_output = gr.Textbox()
            text_button = gr.Button("Prédire")
            text_button.click(predict_images_with_text, inputs=text_input, outputs=text_output)
        with gr.Tab("Recommandation de film par affiche"):
            with gr.Row():
                image_input = gr.Image(label="Déposer votre affiche de film ici : ")
                image_output = [GradioImage(type="pil",label="Film n°"+str(i)+" recommandé : ") for i in range(5)]
            image_button = gr.Button("Prédire")
            refresh_button = gr.Button("Rafraichir")
            image_button.click(predict_images_with_image, inputs=image_input, outputs=image_output)

    

    demo.launch(server_name="0.0.0.0")

