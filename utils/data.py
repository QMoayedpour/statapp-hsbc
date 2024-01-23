import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import trange
import torch.optim as optim
from torch.autograd import grad as torch_grad

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