import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def data_simulation(start=0, phi1=0.8, phi2=-0.2, trend=0.2, amplitude=1, frequency=0.03, sigma=2, n=300, change=True):


    white_noise = np.random.normal(size=n)*sigma

    y_minus_1 = start
    y_minus_2 = 0.0
    tren=trend
    y = [start]
    for t in range(n):
        trend_ = tren * t if t < 200 else tren * 200  # Tendance linéaire

        seasonality = amplitude * np.sin(2 * np.pi * frequency * t)

        ar2_component = phi1 * y_minus_1 + phi2 * y_minus_2

        if 400 <= t < 500 and change:
            trend_ -= trend/2 * (t - 400)
        else:
            trend_ = tren * t 

        y_t = trend_ + seasonality + ar2_component + white_noise[t]
        y.append(y_t)

        y_minus_2 = y_minus_1
        y_minus_1 = y_t
    return np.array(y)

def simple_plot(y, title= "Simulation d\'une série AR(2)."):
    plt.plot(y, label='Série simulée')
    plt.title(title)
    plt.xlabel('Temps')
    plt.ylabel('Valeurs')
    plt.legend()
    plt.show()


def simple_plot_date(date, y, start_year, title="Cours des actifs", asset_name="Actif", ax=None):
    
    dossier_enregistrement = 'img'
    filtered_date = date[date.dt.year >= start_year]
    filtered_y = y[date.dt.year >= start_year]

    if ax is None:
        plt.clf()
        plt.plot(filtered_date, filtered_y, label=asset_name)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Prix')
        plt.xticks(rotation=45)
        plt.legend()
        plt.savefig(os.path.join(dossier_enregistrement, f'{title}.png'), format='png')
    else:
        ax.plot(filtered_date, filtered_y, label=asset_name)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix')
        ax.legend()
        plt.xticks(rotation=45)   
        plt.savefig(os.path.join(dossier_enregistrement, f'{title}.png'), format='png')


def easy_plot(y, title, label_y):
    """
    Cette fonction permet de tracer un graphique simple

    Paramètres:
    y (array-like): Les données à tracer.
    title (str): Le titre du graphique.
    label_y (str): Le label pour l'axe des y.

    """
    plt.plot(y, label=label_y)
    plt.title(title)
    plt.xlabel('Temps')
    plt.ylabel('Valeurs')
    plt.legend()
    plt.show()
