# Traitement de la base de donnees

def set_age_data(data):
    # Completer les donnees manquantes pour les ages
    for sexe in range(2):
        for classe in range(3):
            # Recuperer la mediane en fonction du sexe et de la classe
            somme_age = data[(data['Sex'] == sexe) & (data['Pclass'] == classe+1)]['Age'].dropna()
            mediane_age = somme_age.median()

            # Attribuer l age median aux donnees manquantes
            data.loc[(data.Age.isnull()) & (data.Sex == sexe) & (data.Pclass == classe+1),'Age'] = mediane_age

    # Remplacer les "float" par des "int"
    data['Age'] = data['Age'].astype(int)
    return data

def traitement(data):
    # Retirer les variables qui ne nous interessent pas pour la classification
    data = data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

    # Remplacer les "string" par des "int"
    data.loc[data['Sex']=='male','Sex'] = 1
    data.loc[data['Sex']=='female','Sex'] = 0

    data = set_age_data(data)

    return data
