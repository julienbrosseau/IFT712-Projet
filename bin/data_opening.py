# Ouverture des fichiers de la base de données

import os
import pandas as pd

class DataOpening():

    def __init__(self):
        self.path       = "data"
        self.data_train = "train.csv"
        self.data_test  = "test.csv"
        self.data_ref   = "gender_submission.csv"

    def get_training_data(self):
        # Ouverture et chargement des donnees d entrainements
        return pd.read_csv(os.path.join(self.path, self.data_train))

    def get_testing_data(self):
        # Ouverture et chargement des donnees de tests
        return pd.read_csv(os.path.join(self.path, self.data_test))

    def get_referencing_data(self):
        # Ouverture et chargement des donnees de references
        return pd.read_csv(os.path.join(self.path, self.data_ref))

    def get_head_data(self, data):
        # Retourne l entete de la base de donnees
        return data.head()

    def get_columns_data(self, data):
        # Retourne les colonnes qui composent la base de donnees
        return data.columns

    def get_info_data(self, data):
        # Retroune les informations les variables
        return data.info()

    def get_describe_data(self, data):
        # Retroune la distribution des variables
        return data.describe()

    def get_null_data(self, data):
        # Retourne le nombre de variable null
        return data.isnull().sum()