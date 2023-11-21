
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from tqdm.notebook import tqdm
from annoy import AnnoyIndex

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path


def transform_dataset():
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    normalize = transforms.Normalize(mean, std)
    inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(mean, std)],
    std= [1/s for s in std]
    )

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    dataset = ImageAndPathsDataset('MLP-20M', transform)

    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    return dataloader

def init_model(device):
    mobilenet = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model = nn.Sequential(mobilenet.features,mobilenet.avgpool,nn.Flatten())
    model = model.to(device)
    model.eval()
    return model

# Fonction pour le traitement des images et l'extraction des vecteurs
def extract_features(dataloader,model):
    features_list = []
    paths_list = []

    for x, paths in tqdm(dataloader):
        with torch.no_grad():
            embeddings = model(x.to(device))
            features_list.extend(embeddings.cpu().numpy())
            paths_list.extend(paths)

    df = pd.DataFrame({
        'features': features_list,
        'path': paths_list
    })

    df.head()

    features = np.vstack(features_list)
    return features

def get_annoy_index(features):
    dim = len(features[0])  # La longueur des vecteurs de caractéristiques
    annoy_index = AnnoyIndex(dim, 'angular')

    for i, embedding in enumerate(features):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)

    # Enregistrement de l'index
    annoy_index.save('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Anno_Index/annoy_index.ann')


def main():
    dataloader = transform_dataset()
    model = init_model(device)
    features = extract_features(dataloader, model)
    # Enregistrement des caractéristiques
    np.save('/Users/hugoguilbot/VALDOM/INSA/AIF2024_Guilbot/Projet_AIF/Features/Affiches_features.npy', features)
    get_annoy_index(features)


if __name__ == "__main__":
    main()