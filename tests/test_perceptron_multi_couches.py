# Test du fichier "perceptron_multi_couches.py"

import bin.data_opening as op
import bin.treatment as tr
import bin.perceptron_multi_couches as mlp

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Récupération des données
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
t_test = data_ref["Survived"]

# Classification par perceptron mutli-couches
mlp = mlp.Mlp()

# Entrainement des donnees
mlp.fit(x_train, t_train)
predict_train = mlp.predict(x_train)

# Prediction sur les donnees de tests
predic_test = mlp.predict(x_train)

# Affichage des donnees en fonction de leur classification
# Affichage erreurs pour l'entrainement et les tests
print("Erreur d'entrainement : ", (1 - mlp.score(x_train, t_train)) * 100, "%")
print("Erreur de test : ", (1 - mlp.score(x_test, t_test)) * 100, "%")


print("Meilleur hyperparametre : ", mlp.get_best_param())


# Affichage matrice de confusion
# Matrice de confusion
sns.heatmap(confusion_matrix(t_test, predic_test), annot=True, lw=2, cbar=False)
plt.title("Matrice de confusion")
plt.ylabel("Valeurs réelles")
plt.xlabel("Valeurs prédis")
plt.show()


