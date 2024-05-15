import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller, acf
import os


# Regarder la normalité

def normality_graph(x, actif="name"):

    dossier_enregistrement = 'img'
    plt.clf()
    m, sd = x.mean(), x.std()
    # Tracé de la distribution des rendements log-normaux de l'actif
    sb.distplot(x, hist=True, kde=True, label=f'{actif}', kde_kws={'shade': True, 'linewidth': 3})
    # Création d'un échantillon gaussien avec la même moyenne et écart-type
    sample = np.random.normal(m, sd, size=100000)
    # Tracé de la distribution gaussienne avec la même moyenne et écart-type
    sb.distplot(sample, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 2}, label='Gaussian Distribution')
    # Affichage de la légende
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(dossier_enregistrement, 'Distribution.png'), format='png')
    plt.show()

# qqplot 


def check_autocorel(serie, nlags, alpha, qstat=True, score_min=0.8, display_stats=True):
    result = acf(serie, nlags=nlags, alpha=alpha, qstat=qstat)
    if display_stats:
        print('Autocorrelations: {}'.format(result[0]))
        print('Confidence intervals: {}'.format(result[1]))
        print('Q stats of Ljung Box test: {}'.format(result[2]))
        print('p-values: {}'.format(result[3]))
    score = sum(result[3] < alpha)/(len(result[3])-1)
    if score > score_min:
        return True, score
    return False, score, np.where(result[3] > alpha)