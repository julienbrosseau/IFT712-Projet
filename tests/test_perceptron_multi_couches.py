# Test du fichier "perceptron_multi_couches.py"

import numpy as np
import bin.data_opening as op
import bin.treatment as tr
import bin.perceptron_multi_couches as mlp

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

# Récupération des données
data_train = op.get_training_data()
data_test  = op.get_testing_data()
data_ref   = op.get_referencing_data()

# Traitement des donnees
data_train = tr.traitement(data_train)
data_test  = tr.traitement(data_test)

# Affiliation des donnees
X_train = data_train.drop(["Survived"], axis=1)
Y_train = data_train["Survived"]
X_test  = data_test
Y_test = data_ref["Survived"]

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) # Taille des jeux de données

# Classification par perceptron mutli-couches
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



