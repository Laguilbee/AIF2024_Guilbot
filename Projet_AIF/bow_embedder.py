import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import re
from nltk.corpus import stopwords
from annoy import AnnoyIndex
import numpy as np
import pickle

# Assurez-vous que ces téléchargements ne sont faits qu'une seule fois
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

class MovieSynopsisBoW:
    def __init__(self, dataframe_path):
        self.dataframe_path = dataframe_path
        self.tokenizer = StemTokenizer()
        self.stop_words = self.tokenizer(' '.join(stop_words))
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)

    def load_data(self):
        df = pd.read_csv(self.dataframe_path)
        df = df.dropna(subset=['overview'])
        return df

    def create_bow_embeddings(self, data):
        bow_matrix = self.vectorizer.fit_transform(data)
        #bow_embeddings = bow_matrix.toarray()
        return bow_matrix
    
    # def create_bow_embeddings(self, data):
    #     bow_matrix = self.vectorizer.fit_transform(data)
    #     bow_embeddings = bow_matrix.toarray()
    #     return bow_embeddings

    # def add_embeddings_to_df(self):
    #     df = self.load_data()
    #     embeddings = self.create_bow_embeddings(df['overview'])
    #     df['bow_embeddings'] = embeddings.tolist()
    #     return df

    def add_embeddings_to_df(self):
        df = self.load_data()
        embeddings = self.create_bow_embeddings(df['overview'])
        
        # Stocker les indices et les valeurs non nulles pour chaque vecteur
        df['bow_embeddings'] = [
            (embedding.nonzero()[1], embedding.data) 
            for embedding in embeddings
        ]
        return df


    def save_embeddings(self, output_path):
        df_with_embeddings = self.add_embeddings_to_df()
        df_with_embeddings.to_pickle(output_path)

    def save_vectorizer(self, output_path):
        df = self.load_data()
        self.vectorizer.fit(df['overview'])
        # Utilisez pickle pour enregistrer l'objet vectoriseur
        with open(output_path, 'wb') as file:
            pickle.dump(self.vectorizer, file)

# def get_annoy_index_movies_bow():
#     df = pd.read_pickle('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_embeddings_bow.pkl')
#     dim = len(df['bow_embeddings'][0])  # La longueur des vecteurs de caractéristiques
#     annoy_index = AnnoyIndex(dim, 'angular')

#     for idx in df.index:  # Utiliser les indices du DataFrame
#         embedding = df.loc[idx, 'bow_embeddings']
#         annoy_index.add_item(idx, embedding)

#     annoy_index.build(10)
#     annoy_index.save('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Annoy_Index/annoy_movies_index_bow.ann')
        
def get_annoy_index_movies_bow():
    df = pd.read_pickle('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_index_embeddings_bow.pkl')
    dim = 65322  # La longueur des vecteurs de caractéristiques
    annoy_index = AnnoyIndex(dim, 'angular')

    for idx in df.index:
        # Initialiser un vecteur dense de zéros
        dense_vector = np.zeros(dim)
        # Récupérer les indices et les valeurs
        indices, values = df.at[idx, 'bow_embeddings']
        
        # Remplir le vecteur dense avec les valeurs non nulles
        for col, val in zip(indices, values):
            dense_vector[col] = val

        # Ajouter le vecteur dense à l'index Annoy
        annoy_index.add_item(idx, dense_vector)

    annoy_index.build(10)
    annoy_index.save('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Annoy_Index/annoy_movies_index_bow.ann')

# Exemple d'utilisation
def main():
    embedder = MovieSynopsisBoW('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Movies/movies_metadata.csv')
    #print(embedder.dim)
    #embedder.save_embeddings('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_index_embeddings_bow.pkl')
    #get_annoy_index_movies_bow()
    embedder.save_vectorizer('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Model_Pretrain/vectoriseur.pkl')
    # df = pd.read_pickle('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_index_embeddings_bow.pkl')
    # print(df.loc[0, 'bow_embeddings'])
    #dim = len(df['bow_embeddings'][0]) 
    #print(dim)


if __name__ == "__main__":
    main()
