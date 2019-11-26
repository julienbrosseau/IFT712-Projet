# Classification par SVM (machines à support de vecteurs)

# Source du module "SVM"
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Source du module "GridSearchCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


from sklearn import svm
from sklearn.model_selection import GridSearchCV

class SVM():
    def __init__(self):
        # Initialisation du module
        parameters = {'kernel':('linear', 'rbf'), 'C':range(1, 20 ,1)} #range en test 
        self.svm = GridSearchCV(svm.SVC(gamma='auto'), parameters, cv=15, iid=False)
        
    def fit(self, x_train, y_train):
         # Retourne l'entrainement du modele par rapport aux donnees
         return self.svm.fit(x_train, y_train)
        
    def predict(self, x_train):
        # Retourne la prediction des donnees
        return self.svm.predict(x_train)
    
    def support_vectors(self, x_train):
        # Retourne les vecteurs de support
        return self.svm.support_vectors_(x_train)
    
    def score(self, x_train, t_train):
        #  Retourne la score moyen des donnees en fonction de leur classe
        return self.svm.score(x_train, t_train)
    
    def get_best_param(self):
        # Retourne le meilleur hyperparametre
        return self.svm.best_params_
    
