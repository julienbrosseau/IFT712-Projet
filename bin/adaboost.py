# Classification par Adaboost

# Source du module "AdaBoostClassifier"
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class AdaBoost():
    def __init__(self):
        # Initialisation du module
        grid_parameters = {'n_estimators': range(5, 15, 1), 'learning_rate': np.arange(0.8, 1.8, 0.1)}

        self.adaboost = GridSearchCV(AdaBoostClassifier(
            random_state = 0
        ), grid_parameters, cv=15, iid=False) 
    
    def fit(self, x_tab, t_tab):
        # Retroune l entrainement du modele par rapport aux donnees 
        return self.adaboost.fit(x_tab, t_tab)

    def predict(self, data):
        # Retourne la prediction des donnees
        return self.adaboost.predict(data)

    def score(self, x_tab, t_tab):
        # Retourne la score moyen des donnees en fonction de leur classe
        return self.adaboost.score(x_tab, t_tab)
    
    def get_best_param(self):
        # Retroune le meilleur hyperparametre
        return self.adaboost.best_params_
    