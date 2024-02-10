# Generative Adversarial Network pour la génération de données financières

![Ensae](./img/ensae.png)
![hsbc](./img/hsbc.png)
 
Projet de Statistiques Appliquées, ENSAE 2023/2024

## Génération de données financières

Dans le cadre du développement, de l'estimation ou de l'utilisation de modèles, les données financières servent de support au modèle. On cherche ainsi à déduire des propriétés à partir d'un état du monde observée. On suppose donc que les réalisations d'une série observée sont les résultats d'un phénomène stochastique. Les méthodes statistiques usuelles supposent que les variations d'un actif financier (par exemple), suivent une certaines lois, dont on peut estimer les paramètres. A partir de ces estimations, on peut ensuite simuler et quantifier différentes réalisation de la série à partir des paramètres estimées. Cette approche statistique nécessite de faire un certain nombre d'hypothèse sur la série observée, qui ne sont pas toujours vérifiée. Les méthodes de Machine Learning se caractérisent par des hypothèses moins fortes et une approche plus empiriste. 

## Les modèles GAN

Les modèles [GAN](https://arxiv.org/pdf/1406.2661.pdf) sont des modèles de génération de données. Ces derniers ce sont inscrits dans le paysage de l'IA grâce à leurs performances, particulièrement sur la génération d'image. Les 