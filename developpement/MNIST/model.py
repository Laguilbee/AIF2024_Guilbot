import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*4*4,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # First convolution followed by
        # a relu activation and a max pooling#
        x = F.relu(self.conv1(x))       
        print('couche 1 relu : ',x.shape)
        x = self.pool(x)                
        print('Max pooling 1 : ',x.shape)

        # a relu activation and a max pooling#
        x = F.relu(self.conv2(x)) 
        print('couche 2 relu  : ',x.shape)
        x = self.pool(x)                
        print('Max pooling 2 : ',x.shape)

        # Flatten 
        x = self.flatten(x)
        print(x.shape)

        # Linear
        x = F.relu(self.fc1(x))
        print('Linear 1 : ',x.shape)

        x = F.relu(self.fc2(x))
        print('Linear 2 : ',x.shape)

        x = self.fc3(x)
        print('Linear 3 : ',x.shape)

        return x
    
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x
    

if __name__ == "__main__":
    x = torch.rand(1,28,28).unsqueeze(0)
    print(x.shape)
    model = MNISTNet()
    h = model(x)
    print(h.shape)