# Classification par Random forest

# Source du module "Ramdom forest"
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_estimators=’warn’,
                             criterion=’gini’,
                             max_depth=None,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features=’auto’,
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=None,
                             random_state=None,
                             verbose=0,
                             warm_start=False,
                             class_weight=None)

def fit(x_train, y_train):
     # Retourne l'entrainement du modele par rapport aux donnees
     return clf.fit(x_train, y_train)
    
def predict(x_train):
    # Retourne la prediction des donnees
    return clf.predict(x_train)

def get_best_param(x_train):
    # Retourne les meilleurs paramètres
    return clf.get_best_param(x_train)

def score(x_train, y_train):
    # Retourne les vecteurs de support
    return clf.support_vectors_(x_train, y_train)

