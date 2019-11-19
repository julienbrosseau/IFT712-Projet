# Test du fichier "traitment.py"

import bin.ouverture as op
import bin.traitement as tr

# Ouverture des fichiers
data_opening = op.DataOpening()

data_train = data_opening.get_training_data()
data_test  = data_opening.get_testing_data()

# Traitement des donnees
treatment = tr.Treatment()

data_train = treatment.data_treatment(data_train)
data_test  = treatment.data_treatment(data_test)

#print(op.get_head_data(data_train))
#print(op.get_null_data(data_train))
#print(op.get_null_data(data_test))
#print(op.get_describe_data(data_test))
print(data_opening.get_describe_data(data_train))
