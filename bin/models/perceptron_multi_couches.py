# Classification par perceptron multi-couches

# Source du module "MLPClassifier"
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class Mlp():
    def __init__(self):
        # Initialisation du module
        grid_parameters = {'alpha': np.arange(0, 0.05, 0.01)}

        self.mlp = GridSearchCV(MLPClassifier(
            max_iter=1000
            ), grid_parameters, cv = 15, iid = False)

    def fit(self, x_train, t_train):
        # Retourne l entrainement du modele par rapport aux donnees  
        return self.mlp.fit(x_train, t_train)
    
    def predict(self, x_train):
        # Retourne la prediction des donnees
        return self.mlp.predict(x_train)
    
    def score(self, x_train, t_train):
        # Retourne le score moyen des donnees en fonction de leur classe
        return self.mlp.score(x_train, t_train)
   
    def get_best_param(self):
        # Retourne le meilleur hyperparametre
        return self.mlp.best_params_ 
