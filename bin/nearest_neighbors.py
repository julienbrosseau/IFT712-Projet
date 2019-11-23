# Classification par K-Nearest Neighbors

# Source du module "KNeighborsClassifier"
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNearestNeighbors():
    def __init__(self):
        # Initialisation du module
        grid_parameters = {'n_neighbors': range(1, 20, 1)}

        self.knn = GridSearchCV(KNeighborsClassifier(), grid_parameters, cv=15, iid=False) 
    
    def fit(self, x_tab, t_tab):
        # Retroune l entrainement du modele par rapport aux donnees 
        return self.knn.fit(x_tab, t_tab)

    def predict(self, data):
        # Retourne la prediction des donnees
        return self.knn.predict(data)

    def score(self, x_tab, t_tab):
        # Retourne la score moyen des donnees en fonction de leur classe
        return self.knn.score(x_tab, t_tab)
    
    def get_best_param(self):
        # Retroune le meilleur hyperparametre
        return self.knn.best_params_
    