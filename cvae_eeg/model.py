import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(config.N_CH, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, config.Z_DIM)
        self.fc_logvar = nn.Linear(64, config.Z_DIM)

    def forward(self, x):
        h = self.conv(x)                  # (B, 32, 4)
        h = h.permute(0, 2, 1)           # (B, 4, 32)
        _, (h_n, _) = self.lstm(h)
        h = h_n.squeeze(0)               # (B, 64)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(config.N_CLASSES, 4)
        self.fc = nn.Linear(config.Z_DIM + 4, 64)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.fc_out = nn.Linear(32, config.N_CH)

    def forward(self, z, c):
        c_emb = self.embedding(c)
        zc = torch.cat([z, c_emb], dim=-1)
        h = F.relu(self.fc(zc))
        h = h.unsqueeze(1).repeat(1, config.WIN_LEN, 1)
        h, _ = self.lstm(h)
        out = self.fc_out(h)
        return out.permute(0, 2, 1)       # (B, 5, 4)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar


def loss_fn(x, x_hat, mu, logvar, beta):
    recon = F.mse_loss(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()
