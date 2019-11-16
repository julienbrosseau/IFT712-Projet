# Traitement de la base de donnees

def traitement(data):
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
    # ------------------------------------------------------------------------------

    return data
