"""
Temporary file to test the developed code.
This file is not intended for production use and should be deleted after testing.
It is used to verify the functionality of the code snippets and configurations.

Author: Giacomo Cangi
"""

import numpy as np
import matplotlib.pyplot as plt
import ansys.dpf.core as dpf

print("\n\t... all packages correctly imported ... \n")

class StrainGauge:
    def __init__(self):
        print("\t... StrainGauge class correctly initialized ... \n")

class Murammim:
    def __init__(self):
        print("\t... Constructor class correctly initialized ...\n")

monoaxial_strain_gauge = StrainGauge()
reconstructor = Murammim()

#// FE model navigation ...
model = dpf.Model(r"src_from_nCODE\cantilever_beam.rst")
metadata = model.metadata
print(metadata.time_freq_support)

# model.plot() #// ...plot the model and its mesh without any results

# print(model)

# mesh = model.metadata.meshed_region
# displacement_operator = model.results.displacement()
# fc_out = displacement_operator.outputs.fields_container()
# mesh.plot(fc_out)

# # node_list = [579, 897, 2499]
# node = 579
print("\t--> works until here <-- \n")