# Classification par SVM (machines à support de vecteurs)

# Source du module "SVM"
#https://scikit-learn.org/stable/modules/svm.html#svm-classification

from sklearn import svm


svm.SVC(C=1.0,
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
        verbose=False
        )

