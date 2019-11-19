# Traitement de la base de donnees

class Treatment():

    def __init__(self):
        pass

    def data_treatment(self, data):
        # Retirer les variables qui ne nous interessent pas pour la classification -----
        data = data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)
        # ------------------------------------------------------------------------------

        # Remplacer les "string" par des "int" pour le sexe ----------------------------
        data['Sex'] = data['Sex'].map({'male':1, 'female':0})
        # ------------------------------------------------------------------------------

        # Completer les donnees manquantes pour les ages -------------------------------
        for sexe in range(2):
            for classe in range(3):
                # Recuperer la mediane en fonction du sexe et de la classe
                somme_age = data[(data['Sex'] == sexe) & (data['Pclass'] == classe+1)]['Age'].dropna()
                mediane_age = somme_age.median()

                # Attribuer l'age median aux donnees manquantes en fonction du sexe et de la classe
                data.loc[(data.Age.isnull()) & (data.Sex == sexe) & (data.Pclass == classe+1), 'Age'] = mediane_age

        # Remplacer les "float" par des "int" pour les ages
        data['Age'] = data['Age'].astype(int)

        # Classification par tranche d'ages
        data.loc[ data['Age'] <= 11                      , 'Age'] = 0,  # Enfants
        data.loc[(data['Age'] > 11) & (data['Age'] <= 17), 'Age'] = 1,  # Adolescents
        data.loc[(data['Age'] > 17) & (data['Age'] <= 39), 'Age'] = 2,  # Jeunes
        data.loc[(data['Age'] > 39) & (data['Age'] <= 59), 'Age'] = 3,  # Adultes
        data.loc[ data['Age'] > 59                       , 'Age'] = 4   # Seniors
        # ------------------------------------------------------------------------------

        # Completer les donnees manquantes pour les ports d'embarqations ---------------
        # Assigner le port le plus frequent (apres analyse des donnees) pour les donnees manquantes
        data.loc[data.Embarked.isnull(), 'Embarked'] = 'S'

        # Remplacer les "string" par des "int" pour les ports
        data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2})
        # ------------------------------------------------------------------------------

        # Completer les donnees manquantes pour le prix du billet ----------------------
        # Assigner le tarif le plus frequent (apres analyse des donnees) pour les donnees manquantes
        data.loc[data.Fare.isnull(), 'Fare'] = 7.895800

        # Classification par tranche de prix
        data.loc[ data['Fare'] <= 17                        , 'Fare'] = 0,  # Pauvres
        data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30) , 'Fare'] = 1,  # Modestes
        data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100), 'Fare'] = 2,  # Aises
        data.loc[ data['Fare'] > 100                        , 'Fare'] = 3   # Riches
        # ------------------------------------------------------------------------------

        return data
