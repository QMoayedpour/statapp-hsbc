import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import trange
import torch.optim as optim
from torch.autograd import grad as torch_grad
from torch.utils.data import Dataset, DataLoader
from datetime import date
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

class Data(object):
    def __init__(self, data, n):
        self.data= data
        self.n = n
        self.augment_data= np.array(self.moving_window(self.data, self.n))

    def moving_window(self,x, length):
        return [x[i: i+ length] for i in range(0,(len(x)+1)-length, 10)]
    
    def get_samples(self, G, latent_dim, ts_dim, batch_size, conditional, use_cuda, data):
        noise = torch.randn((batch_size,1,latent_dim))
        idx = np.random.randint(self.augment_data.shape[0], size=batch_size)

        real_samples = self.augment_data[idx, :]
        
        real_samples = np.expand_dims(real_samples, axis=1)
        real_samples = torch.from_numpy(real_samples)
        
        if conditional>0:
            noise[:,:,:conditional] = real_samples[:,:,:conditional]

        if use_cuda:
            noise = noise.cuda()
            real_samples = real_samples.cuda()
            G.cuda()

        y = G(noise)
        y = y.float()
        y = torch.cat((real_samples[:,:,:conditional].float().cpu(),y.float().cpu()), dim=2)

        if use_cuda:
            y = y.cuda()
        return y.float(), real_samples.float()

def plt_progress(real, fake, epoch, path):
    real = np.squeeze(real)
    fake = np.squeeze(fake)
    
    fig, ax = plt.subplots(2,2,figsize=(10, 10))
    ax = ax.flatten()
    fig.suptitle('Data generation, iter:' +str(epoch))
    for i in range(ax.shape[0]):
        ax[i].plot(real[i], color='red', label='Real', alpha =0.7)
        ax[i].plot(fake[i], color='blue', label='Fake', alpha =0.7)
        ax[i].legend()

    plt.savefig(path+'/Iteration_' + str(epoch) + '.png', bbox_inches='tight', pad_inches=0.5)
    plt.clf()

def generate_fake_scenario(input_, true_input, train, amplifier = 1, num = 5):
    for i in range(num):
        conditional = train.conditional
        noise = torch.randn((1, 1, train.latent_dim))*amplifier
        real_samples = torch.from_numpy(input_[:conditional])
        noise[0][0][:conditional] = real_samples[:conditional]
        noise = noise.cuda()
        v = train.G(noise)
        #fixing the conditional part in the output 
        v[0][0][:conditional] = real_samples[:conditional]
        croissance = np.array(v.float().cpu().detach()[0][0])
        fake_line = np.array([true_input[0]] + [true_input[0] * np.prod(1 + croissance[:i+1]) for i in range(40)])
        plt.plot(fake_line)
    plt.plot(true_input[:len(fake_line)], label=f'Vrai donnée', linewidth=2.5, color="red") 
    plt.legend()
    plt.show()

def show_examples(real, fake, size=2):
    real = np.squeeze(real)
    fake = np.squeeze(fake)

    fig, ax = plt.subplots(size,size,figsize=(10, 10))
    ax = ax.flatten()
    fig.suptitle('Data generation, iter:' +str(3))
    for i in range(ax.shape[0]):
        ax[i].plot(real[i], color='red', label='Real', alpha =0.7)
        ax[i].plot(fake[i], color='blue', label='Fake', alpha =0.7)
        ax[i].legend()

    plt.show()

# Reference date used to convert date to integer
referenceDate = date(1900, 1, 1)
 

def ConvertIntToDate(idate: int) -> dt:
    """ Convert integer into python date type

    :param idate: date in integer

    :return: date in python datetime type
    """
    return None if (idate == None) else dt.fromordinal(referenceDate.toordinal() + int(idate) - 2)

class Loader32(Dataset):
    
    def __init__(self, data, length):
        assert len(data) >= length
        self.data = data
        self.length = length
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.length]).reshape(-1, self.length).to(torch.float32)
        
    def __len__(self):
        return max(len(self.data)-self.length, 0)