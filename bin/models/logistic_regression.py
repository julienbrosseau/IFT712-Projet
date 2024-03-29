# Classification par regression logistique

# Source du module "LogisticRegressionCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

from sklearn.linear_model import LogisticRegressionCV
import numpy as np

class LogisticRegression():

    def __init__(self):
        # Initalisation du modele
        self.logistic_regression = LogisticRegressionCV(
            Cs           = np.arange(3, 4, 1e-3),
            cv           = 15,
            random_state = 0,
            solver       = 'lbfgs',
            multi_class  = 'multinomial',
            max_iter     = 1000)

    def fit(self, x_train, t_train):
        # Retroune l'entrainement du modele par rapport aux donnees 
        return self.logistic_regression.fit(x_train, t_train)

    def predict(self, x_train):
        # Retourne la prediction des donnees
        return self.logistic_regression.predict(x_train)

    def score(self, x_train, t_train):
        # Retourne la score moyen des donnees en fonction de leur classe
        return self.logistic_regression.score(x_train, t_train)
    
    def get_best_param(self):
        # Retroune le meilleur hyperparametre
        return self.logistic_regression.C_
    