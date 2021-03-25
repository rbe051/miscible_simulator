"""
Module to solve miscible flow. This module contains the function miscible_flow,
which takes a discretization class and a problem class as arguments. The
discretization class is found at discretization.MiscibleFlow and the problem is found
at problem.Fractured.
"""
import numpy as np
import porepy as pp
import scipy.sparse as sps
import time

import viz


def initiate_variables(disc, s0):
    """
    Initiate AD variables. Given an initial saturation s0, the pressure
    field and pressure variables are calculated.
    """
    gb = disc.problem.gb
    lam_c0 = np.zeros(gb.num_mortar_cells())
    # Initial guess for the pressure and mortar flux
    p0_init = np.zeros(gb.num_cells())
    lam0_init = np.zeros(gb.num_mortar_cells())
    # Define Ad variables
    p, lam = pp.ad.initAdArrays([p0_init, lam0_init])
    # define dofs indices
    p_ix = slice(gb.num_cells())
    lam_ix = slice(gb.num_cells(), gb.num_cells() + gb.num_mortar_cells())
    s_ix = slice(
        gb.num_cells() + gb.num_mortar_cells(),
        2 * gb.num_cells() + gb.num_mortar_cells(),
    )
    lam_c_ix = slice(
        2 * gb.num_cells() + gb.num_mortar_cells(),
        2 * gb.num_cells() + 2 * gb.num_mortar_cells(),
    )
    # Solve with Newton (should converge in 1 or 2 iterations. Is non-linear due to
    # upstream weights)
    p0 = p.val
    lam0 = lam.val
    sol = np.hstack((p.val, lam.val))
    q = np.zeros(gb.num_faces())

    err = np.inf
    newton_it = 0
    sol0 = sol.copy()
    newton_it = 0

    while err > 1e-9:
        newton_it += 1
        q = darcy(disc, p, s0, lam)
        eq_init = pp.ad.concatenate(
            (
                mass_conservation(disc, lam, q, 0),
                coupling_law_p(disc, p, lam, s0),
            )
        )
        err = np.max(np.abs(eq_init.val))
        sol = sol - sps.linalg.spsolve(eq_init.jac, eq_init.val)

#        sol = sol - linear_solvers.amg(eq_init.jac, eq_init.val, sol)
        p.val = sol[p_ix]
        lam.val = sol[lam_ix]

        if newton_it > 20:
            raise RuntimeError('Failed to converge Newton iteration in variable initiation')

    # Now that we have solved for initial condition, initalize full problem
    p, lam, s, lam_c = pp.ad.initAdArrays([p.val, lam.val, s0, lam_c0])
    sol = np.hstack((p.val, lam.val, s.val, lam_c.val))

    q = darcy(disc, p, s, lam)
    return p, lam, q, s, lam_c, sol, p_ix, lam_ix, s_ix, lam_c_ix


def darcy(disc, p, s, lam):
    """
    Calculate the flux from Darcy's law: q = -K * fs(s) * grad( p )
    """
    kw = disc.problem.flow_keyword
    flux = disc.mat[kw]["flux"]
    bound_flux = disc.mat[kw]["bound_flux"]
    bc_val_p = disc.mat[kw]["bc_values"]
    fs = disc.problem.fractional_flow
    mortar2primary = disc.proj["mortar2primary"]
    # find upwind direction
    v = flux * p
    # Return flux
    return (
        (flux * p) * fs(disc.upwind(s, v))
        + bound_flux * bc_val_p
        + bound_flux * mortar2primary * lam
    )


def trace(disc, kw, p, lam):
    """
    Project cell-centered variables to faces by reconstructing the face value.
    """
    trace_p_cell = disc.mat[kw]["trace_cell"]
    trace_p_face = disc.mat[kw]["trace_face"]
    bc_val_p = disc.mat[kw]["bc_values"]
    return trace_p_cell * p + trace_p_face * (lam + bc_val_p)


def mass_conservation(disc, lam, q, t):
    """
    Equation for mass conservation of the fluid: div( q ) - [[lam]] = source
    """
    div = disc.proj["div"]
    mortar2secondary = disc.proj["mortar2secondary"]
    source = np.sum(disc.problem.source(t), axis=0)
    return div * q - mortar2secondary * lam - source


def coupling_law_p(disc, p, lam, s):
    """
    Fluid flux coupling law: lam = - kn * kr(s) / (a / 2) * (p_frac - p_mat)
    """
    kw = disc.problem.flow_keyword
    kn = disc.mat[kw]["kn"]
    fs = disc.problem.fractional_flow

    mortar_volumes = disc.geo["mortar_volumes"]
    secondary2mortar = disc.proj["secondary2mortar"]
    primary2mortar = disc.proj["primary2mortar"]
    mortar2primary = disc.proj["mortar2primary"]

    avg = disc.proj["avg"]
    if isinstance(lam, pp.ad.Ad_array):
        lam_flux = lam.val
    else:
        lam_flux = lam
    primary_flag = (lam_flux > 0).astype(np.int)
    secondary_flag = 1 - primary_flag
    # Calculate the relative permeability by upwinding the concentration
    kr_var = fs(s)
    kr_upw = (
        (secondary2mortar * kr_var) * secondary_flag +
        (primary2mortar * avg * kr_var) * primary_flag
    )
    return lam / kn / mortar_volumes + (
        secondary2mortar * p - primary2mortar * trace(disc, kw, p, mortar2primary * lam)
    ) * kr_upw


def upwind(disc, s, lam, q):
    """
    Upwind discretication of the term div( s * q )
    """
    div = disc.proj["div"]
    avg = disc.proj["avg"]
    secondary2mortar = disc.proj["secondary2mortar"]
    primary2mortar = disc.proj["primary2mortar"]
    mortar2primary = disc.proj["mortar2primary"] 
    mortar2secondary = disc.proj["mortar2secondary"]
    return (div * (disc.upwind(s, q) * q) + disc.mortar_upwind(
        s, lam, div, avg, primary2mortar, secondary2mortar, mortar2primary, mortar2secondary
    ))


def diffusive(disc, s, lam_c):
    """
    Discretization of the diffusive term div( D grad( s ) )
    """
    kw = disc.problem.transport_keyword
    diff = disc.mat[kw]["flux"]
    bc_val_c = disc.mat[kw]["bc_values"]
    bound_diff = disc.mat[kw]["bound_flux"]
    mortar2primary = disc.proj["mortar2primary"]
    mortar2secondary = disc.proj["mortar2secondary"]
    div = disc.proj["div"]
    avg = disc.proj["avg"]
    return (
        div * (diff * s + bound_diff * (mortar2primary * lam_c + bc_val_c))
        - mortar2secondary * lam_c
    )


def transport(disc, lam, lam0, s, s0, lam_c, lam_c0, q, q0, dt, t):
    """
    Concentration transport: ds / dt + div( s * q) - div(D * grad(s)) = 0
    """
    kw = disc.problem.transport_keyword
    diff = disc.mat[kw]["flux"]
    bound_diff = disc.mat[kw]["bound_flux"]
    trace_c_cell = disc.mat[kw]["trace_cell"]
    trace_c_face = disc.mat[kw]["trace_face"]
    bc_val_c = disc.mat[kw]["bc_values"]
    mass_weight = disc.mat[kw]["mass_weight"]

    # Upwind source term
    source = disc.problem.source(t)[0]
    inflow_flag = (source > 0).astype(np.int)
    outflow_flag = (source < 0).astype(np.int)
    inflow = sps.diags(inflow_flag, dtype=np.int)
    outflow = sps.diags(outflow_flag, dtype=np.int)
    return (
        (s - s0) * (mass_weight / dt) +
        upwind(disc, s, lam, q) -
        inflow * source + outflow * s
    )


def coupling_law_c(disc, s, lam_c):
    """
    Diffusive coupling law. Equivalent to the pressure coupling law
    """
    kw = disc.problem.transport_keyword
    Dn = disc.mat[kw]["dn"]
    mortar_volumes = disc.geo["mortar_volumes"]
    secondary2mortar = disc.proj["secondary2mortar"]
    primary2mortar = disc.proj["primary2mortar"]
    mortar2primary = disc.proj["mortar2primary"]
    return (
        lam_c / Dn / mortar_volumes
        + (secondary2mortar * s
           - primary2mortar * trace(disc, kw, s, mortar2primary * lam_c))
    )


def miscible(disc, problem, verbosity=1):
    """
    Solve the coupled problem of fluid flow and component transport.
    The simulator assumes two components. The variables are pressure, p,
    and saturation of component 0, s.

    The governing equations are Darcy's law and mass conservation is solved
    for the fluid flow:
    u = -K*kr(s) grad p,   div u = Q,
    where kr(s) is the relative permeability that depends on saturation.

    The transport of the saturation is governed by the advection and diffusion:
    \partial phi s /\partial t + div (su) -div (D grad s) = Q_s,

    A darcy type coupling is assumed between grids of different dimensions:
    lambda = -kn * kr(s) * (p^lower - p^higher),
    and similar for the diffusivity:
    lambda_c = -D * (s^lower - s^higher).

    Parameters:
    disc (discretization.MiscibleFlow): A discretization class
    problem (problem.Fracture): A problem class

    Returns:
    None

    The solution is exported to vtk.
    """
    simulation_start = time.time()

    # We solve for inital pressure and mortar flux by fixing the temperature
    # to the initial value.
    s0 = problem.initial_concentration()
    gb = problem.gb
    if verbosity > 0:
        print("Initiate variables")
    p, lam, q, s, lam_c, sol, p_ix, lam_ix, s_ix, lam_c_ix = initiate_variables(
        disc, s0
    )
    if verbosity > 0:
        print("Prepare time stepping")
    dt = problem.time_step_param["initial_dt"]
    t = 0
    k = 0

    # Export initial condition
    exporter = pp.Exporter(
        gb,
        problem.time_step_param["file_name"],
        problem.time_step_param["vtk_folder_name"],
        fixed_grid=False,
    )
    updated_grid = False
    viz.split_variables(
        gb,
        [p.val, s.val, problem.fractional_flow(s.val)],
        ["pressure", "saturation", "fs"]
    )
    if problem.write_vtk_for_time(t, k):
        exporter.write_vtu(["pressure", "saturation", "fs"], time_dependent=True, grid=gb)
        times = [0]
    else:
        times = []

    # Prepare time stepping
    time_disc_tot = 0
    time_output_tot = 0
    time_vtk_tot = 0
    time_nwtn_tot = 0
    while t <= problem.time_step_param["end_time"] - dt + 1e-8:
        time_step_start = time.time()
        time_disc = 0
        time_nwtn = 0
        t += dt
        k += 1
        if verbosity > 0:
            print("Solving time step: ", k, " dt: ", dt, " Time: ", t)

        p0 = p.val
        lam0 = lam.val
        s0 = s.val
        lam_c0 = lam_c.val
        q0 = q.val

        err = np.inf
        newton_it = 0
        sol0 = sol.copy()
        while err > 1e-9:
            newton_it += 1
            # Calculate flux
            q = darcy(disc, p, s, lam)
            tic = time.time()
            equation = pp.ad.concatenate(
                (
                    mass_conservation(disc, lam, q, t),
                    coupling_law_p(disc, p, lam, s),
                    transport(disc, lam, lam0, s, s0, lam_c, lam_c0, q, q0, dt, t),
                    coupling_law_c(disc, s, lam_c),
                )
            )
            err = np.max(np.abs(equation.val))
            time_disc += time.time() - tic
            if err < 1e-9:
                break
            tic = time.time()
            if verbosity > 1:
                print("newton iteration number: ", newton_it - 1, ". Error: ", err)

            sol = sol - sps.linalg.spsolve(equation.jac, equation.val)
            time_nwtn += time.time() - tic
            # Update variables
            p.val = sol[p_ix]
            lam.val = sol[lam_ix]

            s.val = sol[s_ix]
            lam_c.val = sol[lam_c_ix]

            if err != err or newton_it > 10 or err > 10e10:
                # Reset
                if verbosity > 0:
                    print("failed Netwon, reducing time step")
                t -= dt / 2
                dt = dt / 2
                p.val = p0
                lam.val = lam0

                s.val = s0
                lam_c.val = lam_c0

                sol = sol0
                err = np.inf
                newton_it = 0

        # Update auxillary variables
        q.val = darcy(disc, p.val, s.val, lam.val)

        if verbosity > 0:
            print("Converged Newton in : ", newton_it - 1, " iterations. Error: ", err)
        if newton_it < 3:
            dt = dt * 1.2
        elif newton_it < 7:
            dt *= 1.1
        dt = problem.sugguest_time_step(t, dt)

        time_output_start = time.time()
        viz.split_variables(
            gb,
            [p.val, s.val, problem.fractional_flow(s.val)],
            ["pressure", "saturation", "fs"]
        )

        time_output_tot += time.time() - time_output_start

        time_vtk_start = time.time()
        if problem.write_vtk_for_time(t, k):
            exporter.write_vtu(
                ["pressure", "saturation", "fs"], time_dependent=True, grid=gb
            )
            times.append(t)
            exporter.write_pvd(timestep=np.array(times))

        time_vtk_tot += time.time() - time_vtk_start
        time_step_time = time.time() - time_step_start
        time_left = time_step_time * (problem.time_step_param["end_time"] - t) / dt

        time_disc_tot += time_disc
        time_nwtn_tot += time_nwtn
        if verbosity > 0:
            print("Time step took: {0:.3f} s".format(time_step_time))
            print("Discretization took: {0:.3f} s".format(time_disc))
            print("Solving linear system took: {0:.3f} s".format(time_nwtn))
            print(
                "Estimated time left: {0:.3f} s ({1:.3f} h)".format(
                    time_left, time_left / 3600
                )
            )
            print("-------------------------------------------------------------------------------\n")

    time_vtk_start = time.time()
    exporter.write_pvd(timestep=np.array(times))
    time_vtk_tot += time.time() - time_output_start

    if verbosity > 0:

        print(
            "\n Finished simulation. It took: {0:.3f} s ({1:.3f} h)".format(
                time.time() - simulation_start, (time.time() - simulation_start) / 3600
            )
        )
        print(
            "Discretization took: {0:.3f} s  ({1:.3f} h)".format(
                time_disc_tot, time_disc_tot / 3600
            )
        )
        print(
            "Solving linear system took: {0:.3f} s ({1:.3f} h)".format(
                time_nwtn_tot, time_nwtn_tot / 3600
            )
        )
        print(
            "Writing output files took: {0:.3f} s ({1:.3f} h)".format(
                time_output_tot, time_output_tot / 3600
            )
        )
        print(
            "Writing VTK files took: {0:.3f} s ({1:.3f} h)".format(
                time_vtk_tot, time_vtk_tot / 3600
            )
        )
