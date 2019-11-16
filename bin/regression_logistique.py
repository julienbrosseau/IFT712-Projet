# Classification par regression logistique

# Source du module "LogisticRegression"
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(
    random_state = 0,
    solver       = 'lbfgs',
    multi_class  = 'multinomial',
    max_iter     = 1000)

def fit(x_train, t_train):
    # Retroune l'entrainement du modele par rapport aux donnees 
    return logistic_regression.fit(x_train, t_train)

def predict(x_train):
    # Retourne la prediction des donnees
    return logistic_regression.predict(x_train)

def predict_proba(x_train):
    # Retourne la probabilite de chaque classe des donnees
    return logistic_regression.predict_proba(x_train)

def score(x_train, t_train):
    # Retourne la score moyen des donnees en fonction de leur classe
    return logistic_regression.score(x_train, t_train)