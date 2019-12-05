# Test du fichier "treatment.py"

import bin.data_opening as op
import bin.treatment as tr

# Ouverture des fichiers
data_opening = op.DataOpening()

data_train = data_opening.get_training_data()
data_test  = data_opening.get_testing_data()

# Traitment des fichiers
treatment = tr.Treatment()

data_train = treatment.data_treatment(data_train)
data_test  = treatment.data_treatment(data_test)

#print(data_opening.get_head_data(data_train))
#print(data_opening.get_null_data(data_train))
#print(data_opening.get_null_data(data_test))
#print(data_opening.get_describe_data(data_test))
print(data_opening.get_describe_data(data_train))
