# Classification par Random forest

# Source du module "Ramdom forest"
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class randomForest():
    def __init__(self):
        #Initialisation du module
        grid_parameters = {'n_estimators': range(1, 20, 1)}

        self.randomForest = GridSearchCV(RandomForestClassifier(
            n_estimators=30, 
            max_depth=2,
            random_state=0
        ), grid_parameters, cv=15, iid=False) 

    def fit(self, x_train, y_train):
        # Retourne l'entrainement du modele par rapport aux donnees
        return self.fit(x_train, y_train)
        
    def predict(self, x_train):
        # Retourne la prediction des donnees
        return self.predict(x_train)
    
    def score(self, x_train, y_train):
        # Retourne les vecteurs de support
        return self.randomForest(x_train, y_train)
    
    def get_best_param(self):
        # Retourne le meilleur hyperparametre
        return self.get_params()
    
