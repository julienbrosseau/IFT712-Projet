# Test du fichier "traitment.py"

import bin.ouverture as op
import bin.traitement as tr

# Traitement des donnees
data_train = op.get_training_data()
data_test  = op.get_testing_data()

data_train = tr.traitement(data_train)

print(op.get_head_data(data_train))
