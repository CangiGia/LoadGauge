"""
Temporary file to test the developed code.
This file is not intended for production use and should be deleted after testing.
It is used to verify the functionality of the code snippets and configurations.

Author: Giacomo Cangi
"""

import numpy as np
import matplotlib.pyplot as plt
import ansys.dpf.core as dpf
from tabulate import tabulate

#// --------------------------------------------
print("\n\t... all packages correctly imported ... \n")

class StrainGauge:
    def __init__(self):
        print("\t... StrainGauge class correctly initialized ... \n")

class Murammim:
    def __init__(self):
        print("\t... Constructor class correctly initialized ...\n")

mono_strain_gauge = StrainGauge()
reconstructor = Murammim()
#// --------------------------------------------

#// FE model navigation ... 
#// --------------------------------------------
print("\t... starting FE model navigation ...\n")

# ---- loading the rst Ansys file and plotting the model 
model = dpf.Model(r"src\giacomo\my_beam.rst")
answer = input("\t\t Do you want to plot the model? (y/n): ").strip().lower()
print(f" ")
if answer == 'y':
    model.plot()

# # ---- defining some scoping parameters, manually for now 
# nodes_to_scope = [579, 897, 2499]
# actual_timestep = 2

# # ---- some available metadata
# nr_time_steps = model.metadata.time_freq_support.n_sets
# available_NS = model.metadata.available_named_selections
# print(f"\t\t Number of time steps: {nr_time_steps} ... \n")
# if available_NS:
#     print(f"\t\t Available named selections: {available_NS} ... \n")
# else:
#     print("\t\t No named selections available ... \n")

# # ---- scoping the nodes and time steps
# nodes_scoping = dpf.Scoping(ids=nodes_to_scope, location=dpf.locations.nodal)
# time_scoping = dpf.time_freq_scoping_factory.scoping_by_load_steps([actual_timestep])

# # ---- operator definition to retrieve strains
# op = dpf.operators.result.elastic_strain()
# # ---- input for operator 
# op.inputs.time_scoping.connect(time_scoping)
# op.inputs.mesh_scoping.connect(nodes_scoping)
# op.inputs.data_sources.connect(model)
# op.inputs.bool_rotate_to_global.connect(True)
# # ---- output from the operator
# out_fc = op.outputs.fields_container()
# field = out_fc[0]  # get the first field from the fields container

# if len(out_fc) > 0:
#         # The array 'field.data' contains the strain values
#         data_values = field.data

#         # The node IDs are in the field's scoping
#         node_ids = field.scoping.ids

#         # Combine node IDs with their strain data
#         table_data = []
#         for i, node_id in enumerate(node_ids):
#             # Each row of the table will be [Node_ID, E_xx, E_yy, E_zz, E_xy, E_yz, E_xz]
#             row = [node_id] + list(data_values[i])
#             table_data.append(row)
            
#         headers = ["Node ID", "E_xx", "E_yy", "E_zz", "E_xy", "E_yz", "E_xz"]
#         print("\t\t Elastic Strain Results @ nodes: ", nodes_to_scope, " @ time step: ", actual_timestep)
#         print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4e", numalign="center", stralign="center"))
#         # - tablefmt="grid": creates a table with well-drawn borders
#         # - floatfmt=".4e": formats numbers in scientific notation with 4 decimal digits
# else:
#     print("\t\t No results found for the given scoping.")
# #// --------------------------------------------

# print("\n\t---> works until here <--- \n")