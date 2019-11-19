# Classification par Ridge

# Source du module "RidgeClassifierCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html

from sklearn.linear_model import RidgeClassifierCV
import numpy as np

class Ridge():
    def __init__(self):
        # Initalisation du modele
        self.ridge = RidgeClassifierCV(
            alphas   = np.array(np.arange(1e-6, 1e-3, 1e-4)),
            cv       = 5
        )

    def fit(self, x_train, t_train):
        # Retroune l entrainement du modele par rapport aux donnees 
        return self.ridge.fit(x_train, t_train)

    def predict(self, data):
        # Retourne la prediction des donnees
        return self.ridge.predict(data)

    def score(self, x_train, t_train):
        # Retourne la score moyen des donnees en fonction de leur classe
        return self.ridge.score(x_train, t_train)

    def get_best_param(self):
        return self.ridge.alpha_