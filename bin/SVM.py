# Classification par SVM (machines à support de vecteurs)

# Source du module "SVM"
#https://scikit-learn.org/stable/modules/svm.html#svm-classification

from sklearn import svm


svm=svm.SVC(C=1.0,
        cache_size=200,
        class_weight=None,
        coef0=0.0,
        decision_function_shape='ovr',
        degree=3, gamma='scale',
        kernel='rbf',
        max_iter=-1,
        probability=False,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False,
        gamma='scale', #verif! cv?
        decision_function_shape='ovo' #verif! cv?
        )


def fit(x_train, y_train):
     # Retroune l'entrainement du modele par rapport aux donnees
     return svm.fit(x_train, y_train)
    
def predict(x_train):
    # Retourne la prediction des donnees
    return svm.predict(x_train)

def support_vectors(x_train):
    # Retourne les vecteurs de support
    return svm.support_vectors_(x_train)