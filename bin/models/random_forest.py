# Classification par Random forest

# Source du module "Ramdom forest"
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class RandomForest():
    def __init__(self):
        #Initialisation du module
        grid_parameters = {'n_estimators': range(15, 20, 1), 'max_depth': range(3, 7, 1)}

        self.randomForest = GridSearchCV(RandomForestClassifier(
            random_state = 0
        ), grid_parameters, cv = 15, iid=False) 

    def fit(self, x_train, y_train):
        # Retourne l'entrainement du modele par rapport aux donnees
        return self.randomForest.fit(x_train, y_train)
        
    def predict(self, x_train):
        # Retourne la prediction des donnees
        return self.randomForest.predict(x_train)
    
    def score(self, x_train, y_train):
        # Retourne les vecteurs de support
        return self.randomForest.score(x_train, y_train)
    
    def get_best_param(self):
        # Retourne le meilleur hyperparametre
        return self.randomForest.best_params_
    
