# -*- coding: utf-8 -*-
"""   
# Godart Arnaud 19 156 869
"""

import ouverture as op
import traitement as tt
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

# Récupération des jeux de données
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
ref_data = pd.read_csv("data/gender_submission.csv")
dataset = [train_data, test_data]

# Traitement des jeux de données
train_data=tt.traitement(train_data)
test_data=tt.traitement(test_data)

# Identification des jeu de données
X_train = train_data.drop(["Survived"], axis=1)
Y_train = train_data["Survived"]
X_test  = test_data
Y_test = ref_data["Survived"]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) # Taille des jeux de données

# Perceptron mutli-couches
mlp = MLPClassifier(hidden_layer_sizes=(25,13,3), max_iter=900)
print(mlp.fit(X_train, Y_train))


train_acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)
print(train_acc_mlp)

# Test d'évaluation de la précision
Y_pred_mlp = mlp.predict(X_test)
print(round(np.sum(np.equal(Y_pred_mlp, Y_test))/len(Y_test) * 100, 2))

# Log perte (Log loss)
print(log_loss(Y_pred_mlp, Y_test))

# Matrice de confusion
print(confusion_matrix(Y_pred_mlp, Y_test, labels=[1,0]))
sns.heatmap(confusion_matrix(Y_pred_mlp, Y_test), annot=True,lw =2,cbar=False)
plt.ylabel("valeurs réelles")
plt.xlabel("Valeurs prédites")
plt.title("matrice de confusion")
plt.show()



