# Classification par Ridge

# Source du module "RidgeClassifierCV"
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html

from sklearn.linear_model import RidgeClassifierCV

ridge = RidgeClassifierCV(
    alphas   = [1e-3, 1e-2, 1e-1, 1],
    cv       = 5
)

def fit(x_train, t_train):
    # Retroune l entrainement du modele par rapport aux donnees 
    return ridge.fit(x_train, t_train)

def predict(data):
    # Retourne la prediction des donnees
    return ridge.predict(data)

def score(x_train, t_train):
    # Retourne la score moyen des donnees en fonction de leur classe
    return ridge.score(x_train, t_train)