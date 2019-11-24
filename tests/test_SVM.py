import numpy as np
import bin.data_opening as op
import bin.treatment as tr
import bin.SVM as svc

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC


# Récupération des données
data_train = op.get_training_data()
data_test  = op.get_testing_data()
data_ref   = op.get_referencing_data()

# Traitement des donnees
data_train = tr.traitement(data_train)
data_test  = tr.traitement(data_test)


# Identification des jeu de données
X_train = data_train.drop(["Survived"], axis=1)
Y_train = data_train["Survived"]
X_test  = data_test
Y_test = data_ref["Survived"]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape) # Taille des jeux de données

# Machine à vecteurs de support
print(svc.fit(X_train, Y_train))

#Evaluation de l'entrainement
train_acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(train_acc_svc)

Y_pred_svc = svc.predict(X_test)

# Test d'évaluation de la précision
print(round(np.sum(np.equal(Y_pred_svc, Y_test))/len(Y_test) * 100, 2))

# Matrice de confusion
print(confusion_matrix(Y_pred_svc, Y_test, labels=[0, 1]))
sns.heatmap(confusion_matrix(Y_test, Y_pred_svc),annot=True,lw =2,cbar=False)
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("CONFUSSION MATRIX VISUALIZATION")
plt.show()
