# Ouverture des fichiers de la base de données

import os
import pandas as pd

path       = "data"
data_train = "train.csv"
data_test  = "test.csv"
data_ref   = "gender_submission.csv"

def get_training_data():
    # Ouverture et chargement des donnees d entrainements
    return pd.read_csv(os.path.join(path, data_train))

def get_testing_data():
    # Ouverture et chargement des donnees de tests
    return pd.read_csv(os.path.join(path, data_test))

def get_referencing_data():
    # Ouverture et chargement des donnees de references
    return pd.read_csv(os.path.join(path, data_ref))

def get_head_data(data):
    # Retourne l entete de la base de donnees
    return data.head()

def get_columns_data(data):
    # Retourne les colonnes qui composent la base de donnees
    return data.columns

def get_info_data(data):
    # Retroune les informations les variables
    return data.info()

def get_describe_data(data):
    # Retroune la distribution des variables
    return data.describe()