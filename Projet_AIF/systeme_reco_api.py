import pickle
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
from annoy import AnnoyIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

#MODEL_PATH = '/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Anno_Index/annoy_index.ann'
MODEL_PATH = '/app/annoy_index.ann'
# Charger votre index Annoy
dim = 576  # Dimension des vecteurs de caractéristiques (ajustez selon votre modèle)
annoy_index = AnnoyIndex(dim, 'angular')
annoy_index.load(MODEL_PATH)

#with open('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/dataframe.pkl', 'rb') as file:
    #df = pickle.load(file)
with open('/app/Dataframe/dataframe.pkl', 'rb') as file:
    df = pickle.load(file)


def search(query_vector, k=5):
    indices = annoy_index.get_nns_by_vector(query_vector, k)
    paths = df['path'][indices]
    return paths

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    vector = data['vector']
    paths = search(vector,5)
    paths_list = paths.tolist()
    return jsonify(paths_list)

if __name__ == '__main__':
    app.run(debug=True)
