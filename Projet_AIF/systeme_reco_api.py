import pickle
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import os
import numpy as np

app = Flask(__name__)

# Chemins vers les fichiers d'index et de données
MODEL_PATH_SYST_RECO = os.getenv('MODEL_PATH_SYST_RECO', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Annoy_Index/annoy_index.ann')
MODEL_PATH_TEXT = os.getenv('MODEL_PATH_TEXT', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Annoy_Index/annoy_movies_index.ann')
DATAFRAME_IMAGE_PATH = os.getenv('DATAFRAME_IMAGE_PATH', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/dataframe.pkl')
DATAFRAME_TEXT_PATH = os.getenv('DATAFRAME_TEXT_PATH', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_embeddings.pkl')

# MODEL_PATH_SYST_RECO = os.getenv('MODEL_PATH_SYST_RECO', '/app/Annoy_Index/annoy_index.ann')
# MODEL_PATH_TEXT = os.getenv('MODEL_PATH_TEXT', '/app/Annoy_Index/annoy_movies_index.ann')
MODEL_PATH_TEXT_BOW = os.getenv('MODEL_PATH_TEXT_BOW', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Annoy_Index/annoy_movies_index_bow.ann')

# DATAFRAME_IMAGE_PATH = os.getenv('DATAFRAME_IMAGE_PATH', '/app/Dataframe/dataframe.pkl')
# DATAFRAME_TEXT_PATH = os.getenv('DATAFRAME_TEXT_PATH', '/app/Dataframe/movies_with_embeddings.pkl')
DATAFRAME_TEXT_PATH_BOW = os.getenv('DATAFRAME_TEXT_PATH_BOW', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_index_embeddings_bow.pkl')

print("------------1---------")
# Chargement des index et des dataframes une seule fois
dim_image = 576 
dim_text = 768 
dim_text_bow = 65322

annoy_index_image = AnnoyIndex(dim_image, 'angular')
annoy_index_text = AnnoyIndex(dim_text, 'angular')
annoy_index_text_bow = AnnoyIndex(dim_text_bow, 'angular')

annoy_index_image.load(MODEL_PATH_SYST_RECO)
annoy_index_text.load(MODEL_PATH_TEXT)
annoy_index_text_bow.load(MODEL_PATH_TEXT_BOW)

with open(DATAFRAME_IMAGE_PATH, 'rb') as file:
    df_image = pickle.load(file)
with open(DATAFRAME_TEXT_PATH, 'rb') as file:
    df_text = pickle.load(file)
with open(DATAFRAME_TEXT_PATH_BOW, 'rb') as file:
    df_text_bow = pickle.load(file)

def rebuild_dense_vector(sparse_indices,sparse_values, dim):
    # Initialiser un vecteur dense de zéros
    dense_vector = np.zeros(dim)
    # Remplir le vecteur dense avec les valeurs non nulles
    for col, val in zip(sparse_indices, sparse_values):
        dense_vector[col] = val
    
    return dense_vector



def search(image_bool,methode_bool, query_vector, k=5):
    try:
        if image_bool:
            indices = annoy_index_image.get_nns_by_vector(query_vector, k)
            paths = df_image['path'][indices]
            return paths
        else:
            if methode_bool :
                indices = annoy_index_text.get_nns_by_vector(query_vector, k)
                print("indices : ",indices)
                titles = df_text['title'][indices]
                print("les titres : ",titles)
                return titles
            else :
                indices = query_vector["indices"]
                values = query_vector["values"]
                dense_vector = rebuild_dense_vector(indices, values, dim_text_bow)
                indices = annoy_index_text_bow.get_nns_by_vector(dense_vector, k)
                titles = df_text_bow['title'][indices]
                print("les titres : ",titles)
                return titles
    except Exception as e:
        print(f"Error in search: {e}")
        return []

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    vector = data['vector']
    image_bool = data['image_bool']
    methode_bool = data['methode_bool']
    print('vector : ',vector)
    if vector is None:
        return jsonify({"error": "No vector provided"}), 400

    results = search(image_bool,methode_bool, vector, 5).tolist()
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5066))
    app.run(debug=True, host='0.0.0.0', port=port)
