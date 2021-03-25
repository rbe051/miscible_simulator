import os
import numpy as np

import porepy as pp


def split_variables(gb, variables, names):
    dof_start = 0
    for g, d in gb:
        dof_end = dof_start + g.num_cells
        for i, var in enumerate(variables):
            if isinstance(var, pp.ad.Ad_array):
                var = var.val
            if d.get(pp.STATE) is None:
                d[pp.STATE] = dict()
            d[pp.STATE][names[i]] = var[dof_start:dof_end]
        dof_start = dof_end


def set_unique_file_name(folder, name, file_extension=".pvd"):
    i = 0
    while i < 1000:
        exists = os.path.isfile(folder + "/" + name + "_run_" + str(i) + file_extension)
        if not exists:
            return name + "_run_" + str(i)
        i += 1
    raise ValueError(
        "Could not set unique file name. Reached maximum value of unique names"
    )
