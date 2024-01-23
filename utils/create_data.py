import pandas as pd
import numpy as np

def data_simulation(phi1=0.8, phi2=-0.2, amplitude=1, frequency= 0.03, sigma= 2, n=300):

    amplitude = 1.0
    frequency = 0.03

    white_noise = np.random.normal(size=n)*sigma

    y_minus_1 = 0.0
    y_minus_2 = 0.0
    tren=0.2
    y = []
    for t in range(n):
        trend = tren * t if t < 200 else tren * 200  # Tendance linéaire

        seasonality = amplitude * np.sin(2 * np.pi * frequency * t)

        ar2_component = phi1 * y_minus_1 + phi2 * y_minus_2

        if 400 <= t < 500:
            trend -= 0.1 * (t - 400)
        if t==499:
            tren=0.15

        y_t = seasonality + ar2_component + white_noise[t]
        y.append(y_t)

        y_minus_2 = y_minus_1
        y_minus_1 = y_t
    return np.array(y)
y = data_simulation(phi1=1.3, phi2=-0.3, sigma=4, n= 10000)

def simple_plot(y):
    plt.plot(y, label='Série simulée')
    plt.title('Simulation d\'une série AR(2).')
    plt.xlabel('Temps')
    plt.ylabel('Valeurs')
    plt.legend()
    plt.show()
simple_plot(y)
