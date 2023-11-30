import pickle
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import os

app = Flask(__name__)

# Chemins vers les fichiers d'index et de données
MODEL_PATH_SYST_RECO = os.getenv('MODEL_PATH_SYST_RECO', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Anno_Index/annoy_index.ann')
MODEL_PATH_TEXT = os.getenv('MODEL_PATH_TEXT', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Anno_Index/annoy_movies_index.ann')
DATAFRAME_IMAGE_PATH = os.getenv('DATAFRAME_IMAGE_PATH', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/dataframe.pkl')
DATAFRAME_TEXT_PATH = os.getenv('DATAFRAME_TEXT_PATH', '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_embeddings.pkl')

# Chargement des index et des dataframes une seule fois
dim_image = 576 
dim_text = 768 

annoy_index_image = AnnoyIndex(dim_image, 'angular')
annoy_index_text = AnnoyIndex(dim_text, 'angular')
annoy_index_image.load(MODEL_PATH_SYST_RECO)
annoy_index_text.load(MODEL_PATH_TEXT)

with open(DATAFRAME_IMAGE_PATH, 'rb') as file:
    df_image = pickle.load(file)
with open(DATAFRAME_TEXT_PATH, 'rb') as file:
    df_text = pickle.load(file)

def search(image_bool, query_vector, k=5):
    try:
        if image_bool:
            indices = annoy_index_image.get_nns_by_vector(query_vector, k)
            paths = df_image['path'][indices]
            return paths
        else:
            indices = annoy_index_text.get_nns_by_vector(query_vector, k)
            titles = df_text['title'][indices]
            return titles
    except Exception as e:
        print(f"Error in search: {e}")
        return []

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    vector = data['vector']
    image_bool = data['image_bool']

    if vector is None:
        return jsonify({"error": "No vector provided"}), 400

    results = search(True, vector, 5).tolist()
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5066))
    app.run(debug=True, host='0.0.0.0', port=port)
