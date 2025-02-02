import torch
import gradio as gr
import torch.nn as nn
from PIL import Image as PILImage
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import requests
from gradio.components import Image as GradioImage
from transformers import DistilBertModel,DistilBertTokenizerFast
import re
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import re
from nltk.corpus import stopwords
import pickle

# Configuration de l'appareil (GPU ou CPU) pour le traitement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètres de normalisation des images pour le modèle MobileNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)

# Transformation appliquée aux images avant de les passer au modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# Téléchargement des données nécessaires pour le traitement du texte
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

# Classe pour tokeniser et lemmatiser le texte
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

# Initialisation des modèles pour les images et le texte
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

# Chargement du vectoriseur pour le traitement de texte BoW
def init_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

# Initialisation des modèles et du vectoriseur
model = init_model(device)
distilBertModel = init_distilbert_model(device)
vectorizer = init_vectorizer('/app/Model/new_vectoriseur.pkl')

# Fonctions pour normaliser les images et extraire les caractéristiques
def normalize_image(image):
    image = PILImage.fromarray(image)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def extract_features(image, model):
    image_tensor = normalize_image(image)
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy()

# Transformation du texte en vecteur BoW
def description_to_bow_vector(description):
    bow_vector = vectorizer.transform([description])
    return bow_vector

# Nettoyage du texte
def clean_text(text):
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir en minuscules
    text = text.lower()
    return text

# Fonctions de prédiction pour les images et le texte
def predict_images_with_image(image):
    feature_vector = extract_features(image, model).flatten()
    #print("feature_vector = ",feature_vector)
    response = requests.post(
        "http://annoy-db:5066/recommend", 
        json={"vector": feature_vector.tolist(),"image_bool":True, "methode_bool": False}
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
        json={"vector": embeddings.tolist(),"image_bool":False,"methode_bool": True}
    )

    print("reponse : ",response)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur API: {response.status_code}")
        return f"Erreur API: {response.status_code}"

def predict_images_with_text_bow(description):
    vector = description_to_bow_vector(description)
    
    vector = vector.toarray()[0]
    print("passage 1",len(vector))
    response = requests.post(
        "http://annoy-db:5066/recommend", 
        json={"vector": vector.tolist(), "image_bool": False, "methode_bool": False}
    )

    print("reponse : ",response)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur API: {response.status_code}")
        return f"Erreur API: {response.status_code}"

# Configuration et lancement de l'interface Gradio
if __name__ == '__main__':

    with gr.Blocks() as demo:
        gr.Markdown("""
        # LE SYSTÈME DE RECOMMANDATION !
        """)
        with gr.Tab("Recommandation de film par synopsis (BERT)"):
            text_input = gr.Textbox(label="Description :",placeholder="Taper la descrition d'un film")
            text_output = gr.Textbox(label="Recommandation :")
            text_button = gr.Button("Prédire")
            clear = gr.Button("Clear")
            text_button.click(predict_images_with_text, inputs=text_input, outputs=text_output)
            clear.click(lambda: None, None, text_input, queue=False)
        with gr.Tab("Recommandation de film par synopsis (BOW)"):
            text_input = gr.Textbox(label="Description :",placeholder="Taper la descrition d'un film")
            text_output = gr.Textbox(label="Recommandation :")
            text_button = gr.Button("Prédire")
            clear = gr.Button("Clear")
            text_button.click(predict_images_with_text_bow, inputs=text_input, outputs=text_output)
            clear.click(lambda: None, None, text_input, queue=False)
        with gr.Tab("Recommandation de film par affiche"):
            with gr.Row():
                image_input = gr.Image(label="Déposer votre affiche de film ici : ")
                image_output = [GradioImage(type="pil",label="Film n°"+str(i)+" recommandé : ") for i in range(5)]
            image_button = gr.Button("Prédire")
            clear = gr.Button("Clear")
            image_button.click(predict_images_with_image, inputs=image_input, outputs=image_output)
            clear.click(lambda: None, None, image_output, queue=False)

    

    demo.launch(server_name="0.0.0.0")

