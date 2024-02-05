import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_simulation(start = 0,phi1=0.8, phi2=-0.2, trend=0.2,amplitude=1, frequency= 0.03, sigma= 2, n=300, change=True):


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
