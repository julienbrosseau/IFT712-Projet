# Test du fichier "data_opening.py"

import bin.data_opening as op

data_opening = op.DataOpening()

# Recuperation des bases de donnees
data_train = data_opening.get_training_data()
data_test  = data_opening.get_testing_data()
data_ref   = data_opening.get_referencing_data()

# Affichage de l entete
head_train = data_opening.get_head_data(data_train)
head_test  = data_opening.get_head_data(data_test)
#print(head_train)

# Affichage des colonnes
columns_train = data_opening.get_columns_data(data_train)
columns_test  = data_opening.get_columns_data(data_test)
#print(columns_train)

# Affichage des informations
#info_train = data_opening.get_info_data(data_train)
#info_test  = data_opening.get_info_data(data_test)

# Affichage de la distibution
describe_train = data_opening.get_describe_data(data_train)
describe_test  = data_opening.get_describe_data(data_test)
#print(describe_train)

# Affichage du nombre de variables null
null_train = data_opening.get_null_data(data_train)
null_test  = data_opening.get_null_data(data_test)
print(null_train)