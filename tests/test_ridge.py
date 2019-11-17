# Test du fichier "ridge.py"

import bin.ouverture as op
import bin.traitement as tr
import bin.ridge as rd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Récupération des donnees
data_train = op.get_training_data()
data_test  = op.get_testing_data()
data_ref   = op.get_referencing_data()

# Traitement des donnees
data_train = tr.traitement(data_train)
data_test  = tr.traitement(data_test)

# Affiliation des donnees
x_train = data_train.drop(["Survived"], axis=1)
t_train = data_train["Survived"]
x_test  = data_test
t_test  = data_ref["Survived"]

# Classification par ridge
# Entrainement des donnees
rd.fit(x_train, t_train)
predic_train = rd.predict(x_train)

# Test de l'entrainement du modele
rd.fit(x_test, t_test)
predic_test = rd.predict(x_test)

# Affichage des donnees en fonction de leur classification
# Affichage erreurs pour l'entrainement et les tests
print("Erreur d'entrainement :", (1 - rd.score(x_train, t_train)) * 100, "%")
print("Erreur de test :", (1 - rd.score(x_test, t_test)) * 100, "%")

# Affichage matrice de confusion
sns.heatmap(confusion_matrix(t_train, predic_train),annot=True,lw =2,cbar=False)
#sns.heatmap(confusion_matrix(t_test, predic_test),annot=True,lw =2,cbar=False)

plt.title("Matrice de confusion")
plt.ylabel("Valeurs réelles")
plt.xlabel("Valeurs prédis")

plt.show()