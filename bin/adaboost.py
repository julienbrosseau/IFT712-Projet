# Classification par Adaboost

# Source du module "AdaBoostClassifier"
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

class AdaBoost():
    def __init__(self):
        # Initialisation du module
        grid_parameters = {'n_estimators': range(50, 60, 1)}

        self.adaboost = GridSearchCV(AdaBoostClassifier(
            random_state = 0
        ), grid_parameters, cv=20, iid=False) 
    
    def fit(self, x_train, t_train):
        # Retroune l entrainement du modele par rapport aux donnees 
        return self.adaboost.fit(x_train, t_train)

    def predict(self, data):
        # Retourne la prediction des donnees
        return self.adaboost.predict(data)

    def score(self, x_train, t_train):
        # Retourne la score moyen des donnees en fonction de leur classe
        return self.adaboost.score(x_train, t_train)
    
    def get_best_param(self):
        # Retroune le meilleur hyperparametre
        return self.adaboost.best_params_
    