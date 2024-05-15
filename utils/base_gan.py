import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, ts_dim, condition, dropout_rate=0.5):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(256, ts_dim - condition),
            #nn.Tanh()  
        )

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, ts_dim, dropout_rate=0.4):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ts_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)
