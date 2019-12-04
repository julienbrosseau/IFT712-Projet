# Classification par perceptron multi-couches

# Source du module "MLPClassifier"
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class Mlp():
    def __init__(self):
        #Initialisation du module
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(25,13,21),
            max_iter=1000
            )

    def fit(self, x_train, t_train):
        # Retourne l entrainement du modele par rapport aux donnees  
        return self.mlp.fit(x_train, t_train)
    
    def predict(self, x_train):
        # Retourne la prediction des donnees
        return self.mlp.predict(x_train)
    
    def score(self, x_train, t_train):
        # Retourne le score moyen des donnees en fonction de leur classe
       return self.mlp.score(x_train, t_train)
   
   # def get_best_param(self):
        # Retourne le meilleur hyperparametre
    #    return self.mlp.best_params
   
    def crossValidation(self):
        # Retourne une cross validation
        grid_parameters = {'hidden_layer_sizes': range(1, 20, 1)}
        GridSearchCV(
            self.mlp, 
            grid_parameters, 
            cv = 15, 
            iid = False
            )
        return self.mlp.crossValidation

