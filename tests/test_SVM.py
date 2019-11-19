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
from sklearn.svm import SVC, LinearSVC


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

# Machine à vecteurs de support
svc = SVC()
print(svc.fit(X_train, Y_train))

#Evaluation de l'entrainement
train_acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(train_acc_svc)

Y_pred_svc = svc.predict(X_test)

# Test d'évaluation de la précision
print(round(np.sum(np.equal(Y_pred_svc, Y_test))/len(Y_test) * 100, 2))

# Matrice de confusion
print(confusion_matrix(Y_pred_svc, Y_test, labels=[0, 1]))
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix(Y_test, Y_pred_svc),annot=True,lw =2,cbar=False)
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("CONFUSSION MATRIX VISUALIZATION")
plt.show()
