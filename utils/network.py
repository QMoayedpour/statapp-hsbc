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

class Generator(nn.Module):
    def __init__(self, latent_dim, ts_dim, condition, dropout_prob=0.5):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim
        self.condition = condition
        self.hidden = 128
        self.dropout_prob = dropout_prob
        
        # Input: (batch_size, 256), Output: (batch_size, 256)
        self.block = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob)
           
        )

        # Input: (batch_size, self.hidden, sequence_length), Output: (batch_size, self.hidden, sequence_length')
        #since we have kernel size 3 and dilation 2, sequence_length' = sequence_length        
        self.block_cnn = nn.Sequential(
            nn.Conv1d(self.hidden,self.hidden, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True),
        )
        self.block_shift = nn.Sequential(
            # Input: (batch_size, self.hidden, sequence_length), Output: (batch_size, 10, sequence_length)
            nn.Conv1d(self.hidden,10, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True),
            # Input: (batch_size, 10, sequence_length), Output: (batch_size, 10*sequence_length)            
            nn.Flatten(start_dim=1),
            # Input: (batch_size, 10*sequence_length), Output: (batch_size, 256)
            nn.Linear(10*self.latent_dim,256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob)
        )
        self.noise_to_latent = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.hidden, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.hidden,self.hidden, kernel_size=5, dilation=2, padding=4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob)
        )
        self.latent_to_output = nn.Sequential(
            nn.Linear(256, self.ts_dim-self.condition),
            nn.Dropout(p=self.dropout_prob)
        )

    #"shortcut connection" to counter vanishing gradients
    def forward(self, input_data):
        x = self.noise_to_latent(input_data)
        x_block = self.block_cnn(x)
        x = x_block +x
        x_block = self.block_cnn(x)
        x = x_block +x
        x_block = self.block_cnn(x)
        x = x_block +x
        x = self.block_shift(x)
        x_block = self.block(x)
        x = x_block + x #torch.cat([x, x_block], 1)
        x_block = self.block(x)
        x = x_block + x #torch.cat([x, x_block], 1)
        x_block = self.block(x)
        x = x_block + x #torch.cat([x, x_block], 1)
        x = self.latent_to_output(x)
        return x[:,None,:]


class Discriminator(nn.Module):
    def __init__(self, ts_dim):
        super(Discriminator,self).__init__()

        self.ts_dim = ts_dim
        self.ts_to_feature = nn.Sequential(
            nn.Linear(self.ts_dim, 512),
            nn.LeakyReLU(inplace=True),
        )
        self.block = nn.Sequential(    
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
        )
        self.to_score = nn.Sequential(
            nn.Linear(512, 1)
        )

            

        

    def forward(self, input_data):

        x = self.ts_to_feature(input_data)
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x = self.to_score(x)
        
        return x


class gen_model():
    def __init__(self, data ,generator, critic, gen_optimizer, critic_optimizer, batch_size, path, ts_dim, latent_dim, D_scheduler, G_scheduler, conditional=3,gp_weight=10,critic_iter=5, n_eval=20, use_cuda=False, _lambda= 1, n=100):
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
        self.diff_mean = []
        self.diff_var = []
        self.score = []
        self.current_score = np.inf
        self._lambda = _lambda
        self.n=n
        self.best_epoch = None


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
                
                d_real = self.D(real_batch)
                d_fake = self.D(fake_batch)

                grad_penalty, grad_norm_ = self._grad_penalty(real_batch, fake_batch)
                # backprop with minimizing the difference between distribution fake and distribution real
                self.D_opt.zero_grad()
                 
                d_loss = d_fake.mean() - d_real.mean() + grad_penalty.to(torch.float32)
                d_loss.backward()
                self.D_opt.step()

                if i == self.critic_iter-1:
                    self.D_scheduler.step()
                    self.losses['LR_D'].append(self.D_scheduler.get_lr())
                    self.losses['D'].append(float(d_loss))
                    self.losses['GP'].append(grad_penalty.item())
                    self.losses['gradient_norm'].append(float(grad_norm_))
                    temp_mean, temp_var = comp_mean_var(self, n=self.n, batch_size = self.batch_size)
                    self.diff_mean.append(temp_mean)
                    self.diff_var.append(temp_var)
                    actual_score = temp_mean + self._lambda * temp_var
                    self.score.append(actual_score)
                    if actual_score < self.current_score:
                        print("best score epoch:", actual_score)
                        torch.save(self.G.state_dict(), self.scorepath + '/gen_'+ f"epoch{epoch}" + '.pt')
                        torch.save(self.D.state_dict(), self.scorepath + '/dis_'  + f"epoch{epoch}" + '.pt') 
                        self.current_score = actual_score
                        self.best_epoch = epoch
            
            self.G_opt.zero_grad()
            fake_batch_critic, real_batch_critic = self.data.get_samples(G=self.G, latent_dim=self.latent_dim, batch_size=self.batch_size, ts_dim=self.ts_dim,conditional=self.conditional,data= self.y, use_cuda=self.use_cuda)
            if self.use_cuda:
                real_batch_critic = real_batch_critic.cuda()
                fake_batch_critic = fake_batch_critic.cuda()
                self.D.cuda()
                self.G.cuda()
            # feed-forward
            d_critic_fake = self.D(fake_batch_critic)

            g_loss =  - d_critic_fake.mean()  # d_critic_real.mean()
            # backprop
            g_loss.backward()
            self.G_opt.step()
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
        self.G.load_state_dict(torch.load(self.scorepath + '/gen_'+ f"epoch{self.best_epoch}" + '.pt'))



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


def comp_mean_var(model,n=10000, batch_size = 50):
    """Tracer la variance des taux de croissances de sous échantillons de la série réelles et générées"

    Args:
        model : model de génération de données
        n (int, optional): nombre de sous échantillon. Defaults to 10000.
        batch_size (int, optional): taille des sous échantillons. Defaults to 50.

    /!\ Cette métrique n'est adapté que pour els modèles cnn et lstm !!!
    """
    train = model
    real_mean = []
    fake_mean = []
    fake_var = []
    real_var = []
    for i in range(n):
        real, fakes = train.data.get_samples(G=train.G, latent_dim=train.latent_dim, batch_size=batch_size, ts_dim=train.ts_dim,conditional=train.conditional,data= train.y, use_cuda=train.use_cuda)
        real_array = real.cpu().detach().numpy().reshape(batch_size,50)
        fake_array = fakes.cpu().detach().numpy().reshape(batch_size,50)
        fake_var.append(np.var(fake_array))
        real_var.append(np.var(real_array))
        fake_mean.append(np.mean(fake_array))
        real_mean.append(np.mean(real_array))

    return abs(np.mean(fake_mean)-np.mean(real_mean)), abs(np.mean(fake_var)-np.mean(real_var))

def comp_mean(model,n=10000, batch_size = 50):
    """Tracer la moyenne des taux de croissances de sous échantillons de la série réelles et générées"

    Args:
        model : model de génération de données
        n (int, optional): nombre de sous échantillon. Defaults to 10000.
        batch_size (int, optional): taille des sous échantillons. Defaults to 50.

    /!\ Cette métrique n'est adapté que pour els modèles cnn et lstm !!!
    """
    train = model
    real_mean = []
    fake_mean = []
    for i in trange(n):
        real, fakes = train.data.get_samples(G=train.G, latent_dim=train.latent_dim, batch_size=batch_size, ts_dim=train.ts_dim,conditional=train.conditional,data= train.y, use_cuda=train.use_cuda)
        real_array = real.cpu().detach().numpy().reshape(batch_size,50)
        fake_array = fakes.cpu().detach().numpy().reshape(batch_size,50)
        fake_mean.append(np.mean(fake_array))
        real_mean.append(np.mean(real_array))
    return np.mean(fake_mean), np.mean(real_mean)
