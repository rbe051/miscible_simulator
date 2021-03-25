import numpy as np
import porepy as pp


class Fractured(object):
    """ Data class for copuled flow and transport.
    """

    def __init__(self, gb, param):
        """
        Parameters:
        mesh_args(dictionary): Dictionary containing meshing parameters.
        """
        self.gb = gb
        self.param = param
        self.time_step_param = param["time_step_param"]
        self.well_pos = param["well_pos"]
        self.well_times = param["well_times"]
        self.well_rates = param["well_rates"]

        g_max = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        (xmin, ymin, zmin) = np.min(g_max.nodes, axis=1)
        (xmax, ymax, zmax) = np.max(g_max.nodes, axis=1)
        self.domain = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
            "L": xmax - xmin,
        }

        self.tol = 1e-8
        self.flow_keyword = "flow"
        self.transport_keyword = "transport"

        self.max_time_step = self.time_step_param['max_dt']

        self.well_cells = []
        for idx in range(self.well_pos.shape[1]):
            pos = np.atleast_2d(self.well_pos[:, idx]).T
            dist = np.sum((pos - self.gb.cell_centers()[:g_max.dim])**2, axis=0)
            self.well_cells.append( np.argmin(dist))

        self.well_cells = np.array(self.well_cells)
        self.add_data()

    def fractional_flow(self, c):
        """
        fractional_flow equals 1/mu(c) = 1 / (exp(-c))
        """
        return pp.ad.exp(-c) ** -1

    def source(self, t):
        """
        Returns the source
        """
        active_cells = (self.well_times[0] <= t) & (self.well_times[1] > t)
        rate = np.zeros((2, self.gb.num_cells()))
        rate[:, self.well_cells[active_cells]] = self.well_rates[:, active_cells]
        return rate

    def add_data(self):
        """
        Add data to the GridBucket
        """
        self.add_flow_data()
        self.add_transport_data()

    def initial_concentration(self):
        """
        Defines the initial condition
        """
        c = np.zeros(self.gb.num_cells())
        return c

    def add_flow_data(self):
        """
        Add the flow data to the grid bucket
        """
        keyword = self.flow_keyword
        # Iterate over nodes and assign data
        for g, d in self.gb:
            param = {}
            # Shorthand notation
            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)

            # Specific volume.
            specific_volume = np.power(
                self.param["aperture"], self.gb.dim_max() - g.dim
            )
            param["specific_volume"] = specific_volume
            # Tangential permeability
            if g.dim == self.gb.dim_max():
                kxx = self.param["km"]
            else:
                kxx = self.param["kf"] * specific_volume

            perm = pp.SecondOrderTensor(kxx * unity)
            param["second_order_tensor"] = perm

            # Source term
            param["source"] = zeros

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)

            param["bc"] = pp.BoundaryCondition(g, bound_faces, "dir")
            param["bc_values"] = bc_val

            pp.initialize_data(g, d, keyword, param)

        # Loop over edges and set coupling parameters
        for e, d in self.gb.edges():
            # Get higher dimensional grid
            g_h = self.gb.nodes_of_edge(e)[1]
            param_h = self.gb.node_props(g_h, pp.PARAMETERS)
            mg = d["mortar_grid"]
            specific_volume_h = (
                np.ones(mg.num_cells) * param_h[keyword]["specific_volume"]
            )
            kn = self.param["kn"] * specific_volume_h / (self.param["aperture"] / 2)
            param = {"normal_diffusivity": kn}
            pp.initialize_data(e, d, keyword, param)

    def add_transport_data(self):
        """
        Add the transport data to the grid bucket
        """
        keyword = self.transport_keyword
        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            param = {}
            d["is_tangential"] = True

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            # Specific volume.
            specific_volume = np.power(
                self.param["aperture"], self.gb.dim_max() - g.dim
            )
            param["specific_volume"] = specific_volume
            # Tangential diffusivity
            if g.dim == self.gb.dim_max():
                kxx = self.param["Dm"] * unity
            else:
                kxx = self.param["Df"] * specific_volume * unity
            perm = pp.SecondOrderTensor(kxx)
            param["second_order_tensor"] = perm

            # Source term
            param["source"] = zeros

            # Mass weight
            param["mass_weight"] = specific_volume * self.param["porosity"] * unity

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)
            if bound_faces.size == 0:
                param["bc"] = pp.BoundaryCondition(g, empty, empty)
            else:
                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces] = 0
                param["bc"] = pp.BoundaryCondition(g, bound_faces, "dir")

            param["bc_values"] = bc_val

            pp.initialize_data(g, d, keyword, param)

        # Normal diffusivity
        for e, d in self.gb.edges():
            # Get higher dimensional grid
            g_h = self.gb.nodes_of_edge(e)[1]
            param_h = self.gb.node_props(g_h, pp.PARAMETERS)
            mg = d["mortar_grid"]
            specific_volume_h = (
                np.ones(mg.num_cells) * param_h[keyword]["specific_volume"]
            )
            dn = self.param["Dn"] * specific_volume_h / (self.param["aperture"] / 2)
            param = {"normal_diffusivity": dn}
            pp.initialize_data(e, d, keyword, param)

    def write_vtk_for_time(self, t, k):
        return True

    def sugguest_time_step(self, t, dt):
        dt = min(dt, self.time_step_param['max_dt'])
        T = self.time_step_param['end_time']
        if np.abs(t - T) > dt / 100:
            dt = min(dt, T - t)
        return dt
