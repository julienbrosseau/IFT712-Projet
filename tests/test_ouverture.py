# Test du fichier "ouverture.py"

import bin.ouverture as op

# Recuperation des bases de donnees
data_train = op.get_training_data()
data_test  = op.get_testing_data()
data_ref   = op.get_referencing_data()

# Affichage de l entete
head_train = op.get_head_data(data_train)
head_test  = op.get_head_data(data_test)
#print(head_train)

# Affichage des colonnes
columns_train = op.get_columns_data(data_train)
columns_test  = op.get_columns_data(data_test)
#print(columns_train)

# Affichage des informations
#info_train = op.get_info_data(data_train)
#info_test  = op.get_info_data(data_test)

# Affichage de la distibution
describe_train = op.get_describe_data(data_train)
describe_test  = op.get_describe_data(data_test)
#print(describe_train)

# Affichage du nombre de variables null
null_train = op.get_null_data(data_train)
null_test  = op.get_null_data(data_test)
print(null_train)