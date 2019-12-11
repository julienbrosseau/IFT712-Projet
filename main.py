# Fichier main du projet
import time
import sys

import bin.running as running

start_time = time.time()

run = running.Running()

# Retourne la methode placee en parametre
method = run.get_method(sys.argv)

if method != None:
    # Retourne l'initalisation du modèle
    classifier = run.get_classifier(method)

    # Récupération des donnees
    data_train, data_test, data_ref = run.set_opening()

    # Traitement des donnees
    data_train, data_test = run.set_treatment(data_train, data_test)

    # Affiliation des donnees
    x_train, t_train, x_test, t_test = run.set_x_tab_t_tab(data_train, data_test, data_ref)

    # Entrainement des donnees
    classifier = run.set_fit_train(classifier, x_train, t_train)

    # Predication sur les donnees de tests
    predict_test = run.set_predict_test(classifier, x_test)

    # Affichage erreurs pour l'entrainement et les tests
    run.get_errors(classifier, x_train, t_train, x_test, t_test)

    # Affichage matrice de confusion
    run.get_confusion_matrix(t_test, predict_test)
    
print("--- %s secondes ---" % (time.time() - start_time))
    