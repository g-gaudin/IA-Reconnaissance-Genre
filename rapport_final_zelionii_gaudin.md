# Rapport final du projet d'IA

par Valentin Zelionii et Guillaume Gaudin

## Rappel de l'objectif du projet

Nous avions réussi à créer et entraîner un réseau de neurones sous Keras grâce à notre set de données de voix anglophones et obtenir une précision de presque 98% dès le premier modèle testé. L'interaction entre le package WarbleR, chargé d'extraire les caractéristiques acoustiques de nos fichiers WAV, et Python ayant été comprise, l'objectif à atteindre est d'être capable en un seul script à extraire les caractéristiques, entraîner le réseau de neurones et lui donner pour analyse les caractéristiques de nos fichiers WAV.

### Choix des technologies :

Langage de programmation : Python version 3.7.5

Bibliothèques logicielles utilisées : 

- WarbleR : une bibliothèque de R pour extraire et analyser des données d'un fichier WAV
- ~~Rpy2~~ : plus nécessaire pour lire nos packages de R (voir Installation de WarbleR)
- Keras : pour la création et l'entrainement de réseaux de neurone.
- TensorFlow : pour le fonctionnement global de Keras et l'apprentissage artificiel.
- Pandas : pour la manipulation et l'analyse de données, notamment des fichiers CSV.
- Scikit-learn : pour la normalisation et la transformation des données.
- Matplotlib : pour obtenir une représentation graphique du gain en performances du réseau.

## Installation et utilisation de WarbleR



## Adaptation du format des données

Le set de données utilisé pour entrainer et tester notre réseau de neurones comporte 21 colonnes :

| meanfreq           | sd                 | median            | Q25                | Q75                | IQR                | skew             | kurt             | sp.ent            | sfm               | meandom   | mindom    | maxdom    | dfrange | modindx | label |
| ------------------ | ------------------ | ----------------- | ------------------ | ------------------ | ------------------ | ---------------- | ---------------- | ----------------- | ----------------- | --------- | --------- | --------- | ------- | ------- | ----- |
| 0.0597809849598081 | 0.0642412677031359 | 0.032026913372582 | 0.0150714886459209 | 0.0901934398654331 | 0.0751219512195122 | 12.8634618371626 | 274.402905502067 | 0.893369416700807 | 0.491917766397811 | 0.0078125 | 0.0078125 | 0.0078125 | 0       | 0       | male  |

Les 20 premières colonnes contiennent les caractéristiques utilisées pour la classification de nos échantillons de voix, la dernière colonne donne le label Homme ou Femme de l'échantillon.

Le résultat fourni par notre script en R est stocké dans un fichier CSV comportant 28 colonnes :

| sound.files | selec | duration          | meanfreq          | sd                 | freq.median | freq.Q25 | freq.Q75 | freq.IQR | time.median        | time.Q25           | time.Q75           | time.IQR           | skew             | kurt             | sp.ent            | time.ent          | entropy           | sfm               | meandom        | mindom         | maxdom         | dfrange       | modindx          | startdom       | enddom         | dfslope            | meanpeakf         |
| ----------- | ----- | ----------------- | ----------------- | ------------------ | ----------- | -------- | -------- | -------- | ------------------ | ------------------ | ------------------ | ------------------ | ---------------- | ---------------- | ----------------- | ----------------- | ----------------- | ----------------- | -------------- | -------------- | -------------- | ------------- | ---------------- | -------------- | -------------- | ------------------ | ----------------- |
| test.wav    | 1     | 0.100454206191639 | 0.191948254683741 | 0.0531175079397912 | 0.17        | 0.16     | 0.19     | 0.03     | 0.0502267573696145 | 0.0334845049130763 | 0.0669690098261527 | 0.0334845049130763 | 1.83899617819243 | 5.27385617610916 | 0.697225815986672 | 0.903685770642495 | 0.630073048831759 | 0.262832143030096 | 0.193798828125 | 0.150732421875 | 0.279931640625 | 0.12919921875 | 1.33333333333333 | 0.279931640625 | 0.193798828125 | -0.857433608461171 | 0.325381168169466 |

Les colonnes {1;2;3;10;11;12;13;17;18;25;26;27;28} du résultat ne seront pas utilisées par notre réseau de neurones. Nous décidons de supprimer les colonnes "mode", "centroid", "meanfun", "minfun", "maxfun" du set d'entrainement car elles ne correspondent à aucune donnée fournie par warbleR, ce qui occasionne une perte non négligeable de 5% de précision, que nous allons essayer de compenser en modifiant le modèle du réseau de neurones.

## Squelette du script générant le réseau de neurones

Tous les projets d'apprentissage artificiel ont une structure similaire ([source](https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5)) :

### Le chargement et la transformation des données :

Notre set de données est un fichier au format CSV contenant plus de 30000 lignes pour une quinzaine de caractéristiques acoustiques. Ce fichier téléchargé depuis ce [site](https://data.world/ml-research/gender-recognition-by-voice) est stocké dans un dossier data de notre projet et lu par la méthode `read_csv()` de pandas :

```python
import pandas as pd
dataset = pd.read_csv('data/voice.csv')
```

Nous utilisons ensuite la méthode `iloc()` pour scinder notre set de données en deux : X contiendra les colonnes des valeurs de caractéristiques acoustiques et y contiendra les sexes "female" ou "male" correspondant. Étant donné l'écart important entre certaines valeurs d'une colonne à une autre, il nous est conseillé de normaliser ces valeurs afin de faciliter l'entrainement du réseau de neurones. Nous utilisons donc l'objet StandardScaler de scikit-learn et réalisons l'opération de normalisation sur X.

```python
from sklearn.preprocessing import StandardScaler
X = dataset.iloc[:, :15].values
y = dataset.iloc[:, 15:16].values
sc = StandardScaler()
X = sc.fit_transform(X)
```

Les catégories "female" ou "male" doivent ensuite être transformées en un vecteur binaire à deux dimensions afin d'être donné à Keras pour réaliser le réseau de neurones :

| male   | 0    | 1    |
| ------ | ---- | ---- |
| female | 1    | 0    |

Nous utilisons l'objet OneHotEncoder de scikit-learn afin de transformer notre tableau y :

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
```

Enfin nous divisons notre set en deux partie : le set d'entrainement du réseau (90%) et le set de test (10%)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
```

### Définition du modèle :

Nous choisissons le modèle séquentiel de construction de réseau de neurones fourni par Keras car il nous semble parfaitement adapté au type de données qu'il devra traiter. La première couche de neurones est forcément de taille 15 car nous avons 15 caractéristiques différentes à traiter. Nous choisissons d'avoir 2 couches intermédiaires, chacune contenant 20 neurones et la couche de sortie 2 neurones pour les catégories "female" ou "male" :

```python
from keras.layers import Dense, Activation
from keras.models import Sequential

# Creation of the neural network
model = Sequential([
    Dense(20, input_dim=15), Activation('relu'),
    Dense(20, input_dim=20), Activation('relu'),
    Dense(2), Activation('sigmoid'), ])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

Les fonctions d'activation 'relu' et 'sigmoid' ont été choisi par défaut et suivant les conseils des différents tutoriels que nous avons suivi.

### Optimisateur, perte et entrainement du modèle :

Nous lançons la compilation du modèle avec un algorithme d'optimisation choisi par défaut en suivant la documentation de Keras. Le modèle défini pour les pertes est imposé par le nombre de catégories de la couche final. L'objectif de l'entrainement sera d'obtenir la meilleure précision possible au bout de 20 itérations, chaque itération étant validée par le set de test :

```python
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=64)
```

## Graphes de précisions et de pertes de chaque itération

Afin d'avoir une visualisation plus évidente de l'évolution de la précision et des pertes de notre réseau de neurones au cours de son entraînement, nous utilisons la bibliothèque Matplotlib :

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

Grâce au choix de notre modèle nous réussissons à atteindre une précision de 95% en seulement 20 itérations :

![](C:\Users\darkg\Pictures\Saved Pictures\precision1.png)

![](C:\Users\darkg\Pictures\Saved Pictures\pertes1.png)