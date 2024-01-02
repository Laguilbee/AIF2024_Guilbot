import pandas as pd
import re
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DistilBertModel,DistilBertTokenizerFast
from tqdm.notebook import tqdm
from annoy import AnnoyIndex
import pickle
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
from annoy import AnnoyIndex
import os

def clean_text(text):
    # Supprimer les caractères spéciaux et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir en minuscules
    text = text.lower()
    return text


class MovieSynopsisEmbedder:

    def __init__(self, dataframe_path, tokenizer_model='distilbert-base-uncased',batch_size=1):
        self.dataframe_path = dataframe_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_model)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DistilBertModel.from_pretrained(tokenizer_model)
        self.model.to(self.device)

    class SynopsisDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data.to_list()
            self.encodings = tokenizer(self.data, truncation=True, padding=True,max_length=512)

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings['input_ids'])

    

    def load_data(self):
        df = pd.read_csv(self.dataframe_path)
        df = df.dropna(subset=['overview'])
        df['overview'] = df['overview'].apply(clean_text)
        return df
    
    def create_embeddings(self, data):
        dataset = self.SynopsisDataset(data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        embeddings = []
        self.model.eval()

        with torch.no_grad():
          for batch in dataloader:
              input_ids = batch['input_ids'].to(self.device)
              attention_mask = batch['attention_mask'].to(self.device)
              outputs = self.model(input_ids, attention_mask=attention_mask)
              last_hidden_states = outputs.last_hidden_state[:, 0, :]
              embeddings.extend(last_hidden_states.cpu().numpy())


        return embeddings

    def add_embeddings_to_df(self):
      df = self.load_data()
      embeddings = self.create_embeddings(df['overview'])
      df['embeddings'] = embeddings
      return df

    def save_embeddings(self, output_path):
      df_with_embeddings = self.add_embeddings_to_df()
      df_with_embeddings.to_pickle(output_path)

def get_annoy_index_movies_text():
    df = pd.read_pickle('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_embeddings.pkl')
    dim = len(df['embeddings'][0])  # La longueur des vecteurs de caractéristiques
    annoy_index = AnnoyIndex(dim, 'angular')

    for idx in df.index:  # Utiliser les indices du DataFrame
        embedding = df.loc[idx, 'embeddings']
        annoy_index.add_item(idx, embedding)

    annoy_index.build(10)
    annoy_index.save('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Anno_Index/annoy_movies_index.ann')


def main():
    # embedder = MovieSynopsisEmbedder('/content/movies_metadata.csv')
    # df_with_embeddings = embedder.add_embeddings_to_df()
    # embedder.save_embeddings('/content/movies_with_embeddings.pkl')
    #get_annoy_index_movies_text()
    # print(pd.__version__)
    # with open('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_embeddings.pkl', 'rb') as file:
    #         df_text = pickle.load(file)
    # #pd.__version__
    # print(df_text)
    df = pd.read_pickle('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Dataframe/movies_with_embeddings.pkl')
    # title = df.loc[25561, 'title']
    # print("title : ",title)
    print(df.loc[0, 'overview'])

if __name__ == "__main__":
    main()
   


