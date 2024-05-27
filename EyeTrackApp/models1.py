import torch
import torch.nn as nn
import torch.nn.functional as F
if False:
    class VAE(nn.Module):
        def __init__(self, latent_dim=32):
            super(VAE, self).__init__()
            
            # Encoder
            self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
            self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.enc_fc1 = nn.Linear(64*15*15, latent_dim)
            self.enc_fc2 = nn.Linear(64*15*15, latent_dim)
            
            # Decoder
            self.dec_fc1 = nn.Linear(latent_dim, 128*15*15)
            self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) 
            self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.dec_conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            
        def encode(self, x):
            x = F.relu(self.enc_conv1(x))
            x = F.relu(self.enc_conv2(x)) 
            x = F.relu(self.enc_conv3(x))
            x = x.view(-1, 64*15*15)
            mu = self.enc_fc1(x)
            log_var = None
            if self.training:
                log_var = self.enc_fc2(x)
            return mu, log_var
            
        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps*std
    
        def decode(self, z):
            x = F.relu(self.dec_fc1(z))
            x = x.view(-1, 128, 15, 15)
            x = F.relu(self.dec_conv1(x))
            x = F.relu(self.dec_conv2(x))
            x = torch.sigmoid(self.dec_conv3(x))
            return x
            
        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var

        def compress(self, x, device='cpu'):
            with torch.no_grad():
                x = x.to(device)
                mu, _ = self.encode(x)
                comp = mu.cpu().numpy()
            return comp

        def reconstruct(self, comp, device='cpu'):
            with torch.no_grad():
                z = torch.from_numpy(comp).to(device)
                recon = self.decode(z)
                recon = recon.cpu().numpy()
            return recon
        
    class Regressor(nn.Module):
        def __init__(self):
            super(Regressor, self).__init__()
            self.vae = VAE()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 16)
            self.fc3 = nn.Linear(16, 2)

        def forward(self, x):
            if self.training:
                mu, var_log = self.vae.encode(x)
                x = self.vae.reparameterize(mu, var_log)
            else:
                x, _ = self.vae.encode(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return torch.tanh(x) * 2
            
        def predict(self, x, device='cpu'):
            with torch.no_grad():
                x = torch.tensor(x).to(device)
                x = self.forward(x)
                comp = x.cpu().numpy()
            return comp
elif True:
    class VAE(nn.Module):
        def __init__(self, latent_dim=32):
            super(VAE, self).__init__()
            
            # Encoder
            self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
            self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.enc_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.enc_fc1 = nn.Linear(64*15*15, latent_dim)
            self.enc_fc2 = nn.Linear(64*15*15, latent_dim)
            
            # Decoder
            self.dec_fc1 = nn.Linear(latent_dim, 128*15*15)
            self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) 
            self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.dec_conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            
        def encode(self, x):
            x = F.relu(self.enc_conv1(x))
            x = F.relu(self.enc_conv2(x)) 
            x = F.relu(self.enc_conv3(x))
            x = x.view(-1, 64*15*15)
            mu = self.enc_fc1(x)
            log_var = None
            if self.training:
                log_var = self.enc_fc2(x)
            return mu, log_var
            
        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps*std
    
        def decode(self, z):
            x = F.relu(self.dec_fc1(z))
            x = x.view(-1, 128, 15, 15)
            x = F.relu(self.dec_conv1(x))
            x = F.relu(self.dec_conv2(x))
            x = torch.sigmoid(self.dec_conv3(x))
            return x
            
        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var

        def compress(self, x, device='cpu'):
            with torch.no_grad():
                x = x.to(device)
                mu, _ = self.encode(x)
                comp = mu.cpu().numpy()
            return comp

        def reconstruct(self, comp, device='cpu'):
            with torch.no_grad():
                z = torch.from_numpy(comp).to(device)  
                recon = self.decode(z)
                recon = recon.cpu().numpy()
            return recon
        
    class Regressor(nn.Module):
        def __init__(self):
            super(Regressor, self).__init__()
            self.vae = VAE()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 16)
            self.fc3 = nn.Linear(16, 2)

        def forward(self, x):
            if self.training:
                mu, var_log = self.vae.encode(x)
                x = self.vae.reparameterize(mu, var_log)
            else:
                x, _ = self.vae.encode(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return torch.tanh(x) * 2
            
        def predict(self, x, device='cpu'):
            with torch.no_grad():
                x = torch.tensor(x).to(device)
                x = self.forward(x)
                comp = x.cpu().numpy()
            return comp