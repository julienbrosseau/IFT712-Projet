# Test du fichier "ridge.py"

import bin.ouverture as op
import bin.traitement as tr
import bin.ridge as rd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Récupération des donnees
data_opening = op.DataOpening()

data_train = data_opening.get_training_data()
data_test  = data_opening.get_testing_data()
data_ref   = data_opening.get_referencing_data()

# Traitement des donnees
treatment = tr.Treatment()

data_train = treatment.data_treatment(data_train)
data_test  = treatment.data_treatment(data_test)

# Affiliation des donnees
x_train = data_train.drop(["Survived"], axis=1)
t_train = data_train["Survived"]
x_test  = data_test
t_test  = data_ref["Survived"]

# Classification par ridge
ridge = rd.Ridge()

# Entrainement des donnees
ridge.fit(x_train, t_train)
predic_train = ridge.predict(x_train)

# Test de l'entrainement du modele
ridge.fit(x_test, t_test)
predic_test = ridge.predict(x_test)

# Affichage des donnees en fonction de leur classification
# Affichage erreurs pour l'entrainement et les tests
print("Erreur d'entrainement :", (1 - ridge.score(x_train, t_train)) * 100, "%")
print("Erreur de test :", (1 - ridge.score(x_test, t_test)) * 100, "%")

print("Meilleur hyperparametre :", ridge.get_best_param())

# Affichage matrice de confusion
sns.heatmap(confusion_matrix(t_train, predic_train),annot=True,lw =2,cbar=False)
#sns.heatmap(confusion_matrix(t_test, predic_test),annot=True,lw =2,cbar=False)

plt.title("Matrice de confusion")
plt.ylabel("Valeurs réelles")
plt.xlabel("Valeurs prédis")

plt.show()