import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import trange
import torch.optim as optim
from torch.autograd import grad as torch_grad
from utils.data import Data, plt_progress
from tqdm import trange

class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, latent_dim, ts_dim,condition, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.ts_dim = ts_dim
        self.condition = condition

        self.lstm = nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, ts_dim- self.condition), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.ts_dim - self.condition)
        return outputs


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element. 

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, ts_dim,n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(ts_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=input.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs


class gen_model_rnn():
    """"Modele pour la génération de données"
    Le code est quasiment identique au précédent, simplement qu'une légère modification doit être fait car si on utilise
    des lstm, l'entrainement classique avec la descente de gradient ne peut pas s'effectuer. On doit donc préciser
    au code "with torch.backends.cudnn.flags(enabled=False):", voici les explications que j'ai trouver:
    https://discuss.pytorch.org/t/when-should-we-set-torch-backends-cudnn-enabled-to-false-especially-for-lstm/106571
    """
    def __init__(self, data ,generator, critic, gen_optimizer, critic_optimizer, batch_size, path, ts_dim, latent_dim, D_scheduler, G_scheduler, conditional=3,gp_weight=10,critic_iter=5, n_eval=20, use_cuda=False):
        self.G = generator
        self.D = critic
        self.G_opt = gen_optimizer
        self.D_opt = critic_optimizer
        self.G_scheduler = G_scheduler
        self.D_scheduler = D_scheduler
        self.batch_size = batch_size
        self.scorepath = path
        self.gp_weight = gp_weight
        self.critic_iter = critic_iter
        self.n_eval = n_eval
        self.use_cuda = use_cuda
        self.conditional = conditional
        self.ts_dim = ts_dim
        self.data = Data(data,ts_dim)
        self.y = data


        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

        
        self.latent_dim = latent_dim
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'LR_G': [], 'LR_D':[]}

    def train_model(self, epochs):
        for epoch in trange(epochs):
            for i in range(self.critic_iter):
                # train the critic
                fake_batch, real_batch = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, batch_size=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional,data= self.y, use_cuda=self.use_cuda)
                if self.use_cuda:
                    real_batch = real_batch.cuda()
                    fake_batch = fake_batch.cuda()
                    self.D.cuda()
                    self.G.cuda()
                with torch.backends.cudnn.flags(enabled=False):
                    d_real = self.D(real_batch)
                    d_fake = self.D(fake_batch)

                    grad_penalty, grad_norm_ = self._grad_penalty(real_batch, fake_batch)
                    # backprop with minimizing the difference between distribution fake and distribution real
                    self.D_opt.zero_grad()
                    
                    d_loss = d_fake.mean() - d_real.mean() + grad_penalty.to(torch.float32)

                    d_loss.backward()
                    
                    self.D_opt.step()
                    

            if self.critic_iter > 0:
                self.D_scheduler.step()
                self.losses['LR_D'].append(self.D_scheduler.get_lr())
                self.losses['D'].append(float(d_loss))
                self.losses['GP'].append(grad_penalty.item())
                self.losses['gradient_norm'].append(float(grad_norm_))
        
            # Train the generator
            self.G_opt.zero_grad()
            fake_batch_critic, _ = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, batch_size=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional,data= self.y, use_cuda=self.use_cuda)
            if self.use_cuda:
                fake_batch_critic = fake_batch_critic.cuda()
                self.D.cuda()
                self.G.cuda()
            # feed-forward
            d_critic_fake = self.D(fake_batch_critic)

            g_loss = -d_critic_fake.mean()  # maximizing the critic's output on fake samples
            # backprop
            with torch.backends.cudnn.flags(enabled=False):  # Désactiver CuDNN temporairement
                g_loss.backward()
            
            self.G_opt.step()
            
            if self.critic_iter == 0:
                self.G_scheduler.step()

            self.losses['LR_G'].append(self.G_scheduler.get_lr())

            # save the loss of feed forward
            self.losses['G'].append(g_loss.item())  # outputs tensor with single value
            if (epoch + 1) % (10*self.n_eval) == 0:
                fake_lines, real_lines = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, batch_size=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional,data= self.y, use_cuda=self.use_cuda)
                self.real_lines = np.squeeze(real_lines.cpu().data.numpy())
                self.fake_lines = np.squeeze(fake_lines.cpu().data.numpy())
                plt_progress(self.fake_lines, self.real_lines,epoch, "./logs")

            fake_lines, real_lines = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, batch_size=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional,data= self.y, use_cuda=self.use_cuda)
            self.real_lines = np.squeeze(real_lines.cpu().data.numpy())
            self.fake_lines = np.squeeze(fake_lines.cpu().data.numpy())
            if (epoch + 1) % 500 ==0:
                name = 'CWGAN-GP_model_Dense3_concat_fx'
                #torch.save(self.G.state_dict(), self.scorepath + '/gen_' + name + '.pt')
                #torch.save(self.D.state_dict(), self.scorepath + '/dis_' + name + '.pt')    



    def _grad_penalty(self, real_data, gen_data):
        batch_size = real_data.size()[0]
        t = torch.rand((batch_size, 1, 1), requires_grad=True)
        t = t.expand_as(real_data)

        if self.use_cuda:
            t = t.cuda()

        # mixed sample from real and fake; make approx of the 'true' gradient norm
        interpol = t * real_data.data + (1-t) * gen_data.data

        if self.use_cuda:
            interpol = interpol.cuda()
        
        prob_interpol = self.D(interpol)
        torch.autograd.set_detect_anomaly(True)
        gradients = torch_grad(outputs=prob_interpol, inputs=interpol,
                               grad_outputs=torch.ones(prob_interpol.size()).cuda() if self.use_cuda else torch.ones(
                                   prob_interpol.size()), create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        #grad_norm = torch.norm(gradients, dim=1).mean()
        #self.losses['gradient_norm'].append(grad_norm.item())

        # add epsilon for stability
        eps = 1e-10
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)
        #gradients = gradients.cpu()
        # comment: precision is lower than grad_norm (think that is double) and gradients_norm is float
        return self.gp_weight * (torch.max(torch.zeros(1,dtype=torch.double).cuda() if self.use_cuda else torch.zeros(1,dtype=torch.double), gradients_norm.mean() - 1) ** 2), gradients_norm.mean().item()
