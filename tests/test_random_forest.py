# Test du fichier "random_forest.py"

import bin.data_opening as op
import bin.treatment as tr
import bin.Random_forest as rf

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Récupération des données
opening = op.DataOpening()
data_train = opening.get_training_data()
data_test  = opening.get_testing_data()
data_ref   = opening.get_referencing_data()

# Traitement des donnees
treatment = tr.Treatment()
data_train = treatment.data_treatment(data_train)
data_test  = treatment.data_treatment(data_test)


# Affiliation des données
x_train = data_train.drop(["Survived"], axis=1)
t_train = data_train["Survived"]
x_test  = data_test
t_test = data_ref["Survived"]


# Classification par random forest
rf = rf.randomForest()

# Entrainement des donnees
rf.fit(x_train, t_train)
predic_train = rf.predict(x_train)

# Prediction sur les donnees de test
predic_test = rf.predict(x_test)

# Affichage des donnees en fonction de leur classification
# Affichage erreurs pour l'entrainement et les tests
print("Erreur d'entrainment : ", (1 - rf.score(x_train, t_train)) * 100, "%")
print("Erreur de test : ", (1 - rf.score(x_test, t_test)) * 100, "%")
print(rf.get_best_param())

# Affichage matrice de confusion
sns.heatmap(confusion_matrix(t_test, predic_test), annot=True, lw =2, cbar=False)

plt.title("Matrice de confusion")
plt.ylabel("Valeurs réelles")
plt.xlabel("Valeurs prédis")
plt.show()