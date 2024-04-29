from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange



def plot_tsne(real_array, fake_array, random=False, num_random=20):
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    num_real = real_array.shape[0]  
    len_real = real_array.shape[2] 
    num_fake = fake_array.shape[0]
    if random:
        array_full = np.concatenate((np.squeeze(real_array), np.squeeze(fake_array), np.random.rand(num_random, len_real)), axis = 0)
        colors = ["red" for i in range(num_real)] + ["blue" for i in range(num_fake)] + ["yellow" for i in range(num_random)] 
    else:
        array_full = np.concatenate((np.squeeze(real_array), np.squeeze(fake_array)), axis = 0)
        colors = ["red" for i in range(num_real)] + ["blue" for i in range(num_fake)] 
    tsne_results = tsne.fit_transform(array_full)
    #colors = ["red" for i in range(num_real)] + ["blue" for i in range(num_fake)] + ["yellow" for i in range(num_real)] 
    f, ax = plt.subplots(1)
        
    plt.scatter(tsne_results[:num_real,0], tsne_results[:num_real,1], 
                c = colors[:num_real], alpha = 0.5, label = "Original")
    plt.scatter(tsne_results[num_real:num_real + num_fake,0], tsne_results[num_real: num_real + num_fake,1], 
                c = colors[num_real:num_real + num_fake], alpha = 0.5, label = "Synthetic")
    if random:
        plt.scatter(tsne_results[num_real+num_fake:,0], tsne_results[num_real*2:,1], 
                c = colors[num_real+num_fake:], alpha = 0.5, label = "Random")

    ax.legend()
        
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()

def plot_var(model,n=10000, batch_size = 50):
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
    for i in trange(n):
        real, fakes = train.data.get_samples(G=train.G, latent_dim=train.latent_dim, batch_size=batch_size, ts_dim=train.ts_dim,conditional=train.conditional,data= train.y, use_cuda=train.use_cuda)
        real_array = real.cpu().detach().numpy().reshape(batch_size,train.ts_dim)
        fake_array = fakes.cpu().detach().numpy().reshape(batch_size,train.ts_dim)
        fake_mean.append(np.var(fake_array))
        real_mean.append(np.var(real_array))

    plt.figure(figsize=(10, 6))
    plt.hist(real_mean, bins=100, color='skyblue', alpha=0.5, label='Real', edgecolor='black')
    plt.hist(fake_mean, bins=100, color='salmon', alpha=0.5, label='Fake', edgecolor='black')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.title('Histogramme de la distribution de la variance des taux de croissances  pour des échantillons pris aléatoirement')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_mean(model,n=10000, batch_size = 50):
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
        real_array = real.cpu().detach().numpy().reshape(batch_size,train.ts_dim)
        fake_array = fakes.cpu().detach().numpy().reshape(batch_size,train.ts_dim)
        fake_mean.append(np.mean(fake_array))
        real_mean.append(np.mean(real_array))

    plt.figure(figsize=(10, 6))
    plt.hist(real_mean, bins=100, color='skyblue', alpha=0.5, label='Real', edgecolor='black')
    plt.hist(fake_mean, bins=100, color='salmon', alpha=0.5, label='Fake', edgecolor='black')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.title('Histogramme de la distribution du taux de croissance moyen sur des échantillons sélectionnés aléatoirement')
    plt.legend()
    plt.grid(True)
    plt.show()
