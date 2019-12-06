# Fonctions pour l'execution du programme
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import bin.data_opening as op
import bin.treatment as tr

import bin.adaboost as adaboost
# TODO: A COMPLETER

class Running():
    
    def __init__(self):
        # Initialisation de la classe
        pass

    def get_method(self, method):
        # Retourne la methode placee en parametre
        list_methods = ["logistic", "ridge", "adaboost", "nearestneighbors", "svm", "perceptron", "randomforest"]

        if method not in list_methods:
            print(
                "\n--------------------- /!\ ---------------------\n",
                "Veuillez entrer l'une des méthodes suivantes :\n",
                "'logistic' ......... 'logistic regression'\n",
                "'ridge' ............ 'ridge'\n",
                "'adaboost' ......... 'adaboost'\n",
                "'nearestneighbors' . 'k-nearest neighbors'\n",
                "'svm' .............. 'svm'\n",
                "'perceptron' ....... 'multi layer perceptron'\n",
                "'randomforest' ..... 'random forest'\n"
                "--------------------- /!\ ---------------------\n"
            )
            return
        else:
            print(
                "\nVous avez entré la méthode :", method, "\n"
            )
            return method

    def get_classifier(self, method):
        # Retourne l'initalisation du modèle
        if method == "logistic":
            classifier = logistic.LogisticRegression()
        elif method == "ridge":
            classifier = ridge.Ridge()
        elif method == "adaboost":
            classifier = adaboost.AdaBoost()
        elif method == "nearestneighbors":
            classifier = knn.KNearestNeighbors()
        else:
            raise RuntimeError()
        # TODO: A COMPLETER
        
        return classifier

    def set_opening(self):
        # Récupération des donnees
        data_opening = op.DataOpening()

        data_train = data_opening.get_training_data()
        data_test  = data_opening.get_testing_data()
        data_ref   = data_opening.get_referencing_data()

        return data_train, data_test, data_ref

    def set_treatment(self, data_train, data_test):
        # Traitement des donnees
        treatment = tr.Treatment()

        data_train = treatment.data_treatment(data_train)
        data_test  = treatment.data_treatment(data_test)

        return data_train, data_test

    def set_x_tab_t_tab(self, data_train, data_test, data_ref):
        # Affiliation des donnees
        x_train = data_train.drop(["Survived"], axis=1)
        t_train = data_train["Survived"]
        
        x_test  = data_test
        t_test  = data_ref["Survived"]

        return x_train, t_train, x_test, t_test

    def set_fit_train(self, classifier, x_train, t_train):
        # Entrainement des donnees
        classifier.fit(x_train, t_train)

        return classifier

    def set_predict_test(self, classifier, x_test):
        # Predication sur les donnees de tests
        predic_test = classifier.predict(x_test)

        return predic_test

    def get_errors(self, classifier, x_train, t_train, x_test, t_test):
        # Affichage erreurs pour l'entrainement et les tests
        print("Erreur d'entrainement :", (1 - classifier.score(x_train, t_train)) * 100, "%")
        print("Erreur de test :", (1 - classifier.score(x_test, t_test)) * 100, "%")

        print("Meilleur hyperparametre :", classifier.get_best_param())

    def get_confusion_matrix(self, t_test, predic_test):
        # Affichage matrice de confusion
        sns.heatmap(confusion_matrix(t_test, predic_test), annot=True, lw=2, cbar=False)

        plt.title("Matrice de confusion")
        plt.ylabel("Valeurs réelles")
        plt.xlabel("Valeurs prédis")

        plt.show()