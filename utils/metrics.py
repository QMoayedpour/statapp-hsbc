from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from scipy.stats import kurtosis, skew
import torch
import random


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


def get_moments(model,n=1000, batch_size = 50, reducer=1):
    """Calculer les estimateurs des 4 premiers moments sur les séries générées et réelles"

    Args:
        model : model de génération de données
        n (int, optional): nombre de sous échantillon. Defaults to 10000.
        batch_size (int, optional): taille des sous échantillons. Defaults to 50.


    """
    train = model
    real_mean = []
    fake_mean = []
    fake_var = []
    real_var = []
    real_skew = []
    real_kurt = []
    fake_skew = []
    fake_kurt = []
    for i in trange(n):
        fakes, real = train.data.get_samples(G=train.G, latent_dim=train.latent_dim, batch_size=batch_size, ts_dim=train.ts_dim,conditional=train.conditional,data= train.y, use_cuda=train.use_cuda)
        real_array = real.cpu().detach().numpy().reshape(batch_size,train.ts_dim)
        fake_array = fakes.cpu().detach().numpy().reshape(batch_size,train.ts_dim)/reducer
        fake_var.append(np.var(fake_array))
        real_var.append(np.var(real_array))
        fake_mean.append(np.mean(fake_array))
        real_mean.append(np.mean(real_array))
        fake_skew.append(skew(fake_array))
        real_skew.append(skew(real_array))
        fake_kurt.append(kurtosis(fake_array))
        real_kurt.append(kurtosis(real_array))
    return {"real mean":np.mean(real_mean), "fake mean": np.mean(fake_mean), "real var":np.mean(real_var), "fake var":np.mean(fake_var),"real skew": np.mean(real_skew), "fake skew": np.mean(fake_skew),
    "real kurtosis":np.mean(real_kurt), "fake kurtosis": np.mean(fake_kurt)}


def evaluate_fake_scenario(input_, true_input, train, n=500,amplifier = 1, num = 5, reducer=5, no_print=False):
    """Compte le nombre de fois en moyenne ou le vrai scénario est en dehors des intervalles de faux scénarios générés

    Args:
        input_ (array): Array de la série des logs returns
        true_input (array): Array de la vraie série
        train (Model): Modèle de génération de données
        n (int, optional): Nombre de fois que évalue le modèle. Defaults to 500.
        amplifier (int, optional): Amplifier le bruit du générateur. Defaults to 1.
        num (int, optional): nombre de séries générées à chaque évaluation. Defaults to 5.
        reducer (int, optional): Réduire l'amplitude des séries générées. Defaults to 5.
        no_print (bool, optional): affichage ou non du résultat. Defaults to False.

    Returns:
        int: score du modèle
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_count = 0
    pb = trange(n, leave=False)
    for j in pb:
        start = random.randint(0, 2000)
        in_ = input_[start:]
        true_in_ = true_input[start:]
        big_arr = np.empty((train.ts_dim))

        noise = torch.randn((num, 1, train.latent_dim)) * amplifier
        real_samples = torch.from_numpy(input_[:train.conditional])
        noise[:, :, :train.conditional] = real_samples
        

        noise = noise.to(device)
        v = train.G(noise) / reducer
        v[:, :, :train.conditional] = real_samples
        croissance = np.array(v.float().cpu().detach()[:, 0,  :])
        fake_lines = np.array([[true_input[0]] + [true_input[0] * np.prod(1 + croissance[i, :j+1]) for j in range(v.shape[2])] for i in range(num)])
        
        # Calcul du nombre de fois que la vraie série est en dehors de l'intervalle
        x = fake_lines 
        min_x = np.min(x, axis=0)
        max_x = np.max(x, axis=0)
        y = true_input[:train.ts_dim+1-train.conditional]
        count = np.sum((y[11:] < min_x[11:]) | (y[11:] > max_x[11:]))
        pb.set_description(f"Nombre d'erreur : {count}")
        total_count += count
    if not no_print:
        print("-"*100,f"\nMoyenne du nombre de fois que la vraie série est sortie de l'intervalle sur {n} simulations pour {num} scénarios simulés :\n", total_count/n, "\n", "-"*100)
    return total_count/n