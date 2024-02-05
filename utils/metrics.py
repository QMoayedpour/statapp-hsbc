from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np



def plot_tsne(real_array, fake_array, random=False):
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    num_real = real_array.shape[0]  
    len_real = real_array.shape[2] 
    num_fake = fake_array.shape[0]
    if random:
        array_full = np.concatenate((np.squeeze(real_array), np.squeeze(fake_array), np.random.rand(num_real, len_real)), axis = 0)
        colors = ["red" for i in range(num_real)] + ["blue" for i in range(num_fake)] + ["yellow" for i in range(num_real)] 

    else:
        array_full = np.concatenate((np.squeeze(real_array), np.squeeze(fake_array)), axis = 0)
        colors = ["red" for i in range(num_real)] + ["blue" for i in range(num_fake)] 
    tsne_results = tsne.fit_transform(array_full)
    colors = ["red" for i in range(num_real)] + ["blue" for i in range(num_fake)] + ["yellow" for i in range(num_real)] 
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