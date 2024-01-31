import pickle
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import os
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration des chemins vers les fichiers d'index et de données
# Ces chemins peuvent être configurés via des variables d'environnement
MODEL_PATH_SYST_RECO = os.getenv('MODEL_PATH_SYST_RECO', '/app/Annoy_Index/annoy_index.ann')
MODEL_PATH_TEXT = os.getenv('MODEL_PATH_TEXT', '/app/Annoy_Index/annoy_movies_index.ann')
MODEL_PATH_TEXT_BOW = os.getenv('MODEL_PATH_TEXT_BOW', '/app/Annoy_Index/annoy_movies_bow.ann')

DATAFRAME_IMAGE_PATH = os.getenv('DATAFRAME_IMAGE_PATH', '/app/Dataframe/dataframe.pkl')
DATAFRAME_TEXT_PATH = os.getenv('DATAFRAME_TEXT_PATH', '/app/Dataframe/movies_with_embeddings.pkl')
DATAFRAME_TEXT_PATH_BOW = os.getenv('DATAFRAME_TEXT_PATH_BOW', '/app/Dataframe/movies_with_embeddings_bow.pkl')

# Dimensions pour les index Annoy (correspondant aux embeddings)
dim_image = 576  # Dimension des embeddings d'images
dim_text = 768   # Dimension des embeddings de texte (BERT)
dim_text_bow = 5000 # Dimension des embeddings de texte (BoW)

# Initialisation des index Annoy pour les différents types de données
annoy_index_image = AnnoyIndex(dim_image, 'angular')
annoy_index_text = AnnoyIndex(dim_text, 'angular')
annoy_index_text_bow = AnnoyIndex(dim_text_bow, 'angular')

# Chargement des index Annoy depuis les fichiers
annoy_index_image.load(MODEL_PATH_SYST_RECO)
annoy_index_text.load(MODEL_PATH_TEXT)
annoy_index_text_bow.load(MODEL_PATH_TEXT_BOW)

# Chargement des dataframes contenant les données (images, textes BERT, textes BoW)
with open(DATAFRAME_IMAGE_PATH, 'rb') as file:
    df_image = pickle.load(file)
with open(DATAFRAME_TEXT_PATH, 'rb') as file:
    df_text = pickle.load(file)
with open(DATAFRAME_TEXT_PATH_BOW, 'rb') as file:
    df_text_bow = pickle.load(file)

# Fonction pour rechercher des recommandations en utilisant Annoy
def search(image_bool,methode_bool, query_vector, k=5):
    try:
        if image_bool:
            # Recherche pour les images
            indices = annoy_index_image.get_nns_by_vector(query_vector, k)
            paths = df_image['path'][indices]
            return paths
        else:
            # Recherche pour le texte (BERT ou BoW)
            if methode_bool :
                indices = annoy_index_text.get_nns_by_vector(query_vector, k)
                titles = df_text['title'][indices]
                return titles
            else :
                indices = annoy_index_text_bow.get_nns_by_vector(query_vector, k)
                titles = df_text_bow['title'][indices]
                return titles
    except Exception as e:
        print(f"Error in search: {e}")
        return []
    
# Route Flask pour gérer les requêtes de recommandation
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    vector = data['vector']
    image_bool = data['image_bool']
    methode_bool = data['methode_bool']

    if vector is None:
        return jsonify({"error": "No vector provided"}), 400

    results = search(image_bool,methode_bool, vector, 5).tolist()
    return jsonify(results)

# Point d'entrée principal pour exécuter l'application Flask
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5066))
    app.run(debug=True, host='0.0.0.0', port=port)
