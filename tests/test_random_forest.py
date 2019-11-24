# Test du fichier "random_forest.py"

import numpy as np
import bin.data_opening as op
import bin.treatment as tr
import bin.random_forest as rf

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Récupération des données
data_train = op.get_training_data()
data_test  = op.get_testing_data()
data_ref   = op.get_referencing_data()

# Traitement des donnees
data_train = tr.treatment(data_train)
data_test  = tr.treatment(data_test)

# Affiliation des donnees
X_train = data_train.drop(["Survived"], axis=1)
Y_train = data_train["Survived"]
X_test  = data_test
Y_test = data_ref["Survived"]


# Classification par random forest
rf.fit(X_train, Y_train)  

print(rf.clf.feature_importances_)
predict_test = rf.predict(X_test)

print("Meilleur hyperparametre :", rf.get_best_param())

# Affichage matrice de confusion
sns.heatmap(confusion_matrix(Y_train, predict_test), annot=True, lw=2, cbar=False) #Verif!

plt.title("Matrice de confusion")
plt.ylabel("Valeurs reelles")