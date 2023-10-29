#autoencoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F

input_image_size = 224  # Adjust according to your input image size

class Autoencoder1(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder2(nn.Module):
    def __init__(self, latent_dim, **kwargs):
        super(Autoencoder2, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # Added layer
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder3(nn.Module):
    def __init__(self):
        super(Autoencoder3, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(3, 512, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  
        self.dec2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.out = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)        
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = (self.bn1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = (self.bn2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = (self.bn3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = (self.bn4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        x = F.relu(self.dec1(x))
        x = (self.bn4(x))
        x = F.relu(self.dec2(x))
        x = (self.bn3(x))
        x = F.relu(self.dec3(x))
        x = (self.bn2(x))
        x = F.relu(self.dec4(x))
        x = (self.bn1(x))
        x = torch.sigmoid(self.out(x))
        return x




