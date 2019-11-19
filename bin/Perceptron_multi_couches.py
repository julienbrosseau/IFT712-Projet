# Classification par perceptron multi-couches

# Source du module "MLPClassifier"
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
        hidden_layer_sizes=(25,13,3),
        max_iter=1000
        )

def fit(x_train, t_train):
    # Retroune l'entrainement du modele par rapport aux donnees 
    return mlp.fit(x_train, t_train)

def predict(x_train):
    # Retourne la prediction des donnees
    return mlp.predict(x_train)

def score(x_train, t_train):
    # Retourne la score moyen des donnees en fonction de leur classe
    return mlp.score(x_train, t_train)