import math
import os
import sys

import meep.adjoint as mpa
import numpy as np
from autograd import grad
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy import signal, special

import meep as mp
import nlopt  # need install nlopt

# np.random.seed(0)
cur_path = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
opt_path = cur_path + "/opt_2/"
back_path = cur_path + "/back/"

if not os.path.exists(back_path):
    os.makedirs(back_path)

# init_file
evaluation_histary_files = []
epses = []
eps_files = []
eps_file = opt_path + "final_eps.npy"
eps = np.load(eps_file)
epses.append(eps)
# wg_gap数量与eps数量相同
wg_gap = []
wg_gap.append(0)
wg_gap.append(0)
# total_eps = np.concatenate(epses, axis=0)


def sweep_rot_angle(rot_angle=25):
    # if True compute flux (Gaussian source), if False compute DFT (Continue source)
    compute_flux = True
    # size of input and output waveguide
    w = 0.5
    h = 0.2

    # resolution size
    grid = 0.02
    resolution = 1 / grid
    design_res = resolution

    # thickness of PML
    dpml = 1

    ##################### length parameter ################
    input_wvg_length = 5
    output_wvg_length = 5
    # design_region
    design_region_x = np.zeros(len(epses))
    for ii in range(0, len(epses)):
        design_region_x[ii] = len(epses[ii]) * (1 / design_res)
    design_region_y = 0.2
    sx = (
        input_wvg_length
        + np.sum(design_region_x)
        + np.sum(wg_gap)
        + output_wvg_length
        + 2 * dpml
    )

    ##################### height parameter ################
    Substrate_thickness = 0.5
    BOX_thickness = 2
    TOX_thickness = 0.7
    near_field_height = 1
    sy = (
        BOX_thickness
        + TOX_thickness
        + Substrate_thickness
        + near_field_height
        + 2 * dpml
    )

    plane_y = 0.5 * sy - dpml - 0.5 * near_field_height
    plane_x = (
        -0.5 * sx
        + input_wvg_length
        + 0.5 * (np.sum(design_region_x) + np.sum(wg_gap))
        + dpml
    )
    source_x = np.sum(design_region_x) + 3

    cell = mp.Vector3(sx, sy, 0)

    # Material setting
    Si = mp.Medium(index=3.45)
    SiO2 = mp.Medium(index=1.44)
    SiN = mp.Medium(index=1.96)

    ####################### geometry ##########################
    Substrate_geo = mp.Block(
        material=Si,
        center=mp.Vector3(y=-0.5 * sy + 0.5 * Substrate_thickness + dpml),
        size=mp.Vector3(sx, Substrate_thickness),
    )
    BOX_geo = mp.Block(
        material=SiO2,
        center=mp.Vector3(
            y=-0.5 * sy + Substrate_thickness + 0.5 * BOX_thickness + dpml
        ),
        size=mp.Vector3(sx, BOX_thickness),
    )
    TOX_geo = mp.Block(
        material=SiO2,
        center=mp.Vector3(
            y=-0.5 * sy
            + Substrate_thickness
            + BOX_thickness
            + 0.5 * TOX_thickness
            + dpml
        ),
        size=mp.Vector3(sx, TOX_thickness),
    )
    input_waveguide_geo = mp.Block(
        material=SiN,
        center=mp.Vector3(
            -0.5 * sx + 0.5 * input_wvg_length,
            -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
        ),
        size=mp.Vector3(input_wvg_length + 2 * dpml, h),
    )
    output_waveguide_geo = mp.Block(
        material=SiN,
        center=mp.Vector3(
            0.5 * sx - 0.5 * output_wvg_length,
            -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
        ),
        size=mp.Vector3(output_wvg_length + 2 * dpml, h),
    )

    geometry = [
        Substrate_geo,
        BOX_geo,
        TOX_geo,
        input_waveguide_geo,
        output_waveguide_geo,
    ]

    ####################### geometry ##########################

    # ####################### design region ##########################
    for ii in range(0, len(epses)):
        design_region_resolution_x = design_res
        design_region_resolution_y = 5  # 200nm resolution
        Nx = int(design_region_resolution_x * design_region_x[ii])
        Ny = int(design_region_resolution_y * design_region_y)
        design_variables = mp.MaterialGrid(
            mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_DEFAULT"
        )
        design_region = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
                center=mp.Vector3(
                    -0.5 * sx
                    + input_wvg_length
                    + 0.5 * design_region_x[ii]
                    + dpml
                    + np.sum(design_region_x[0:ii])
                    + np.sum(wg_gap[0:ii]),
                    -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
                    0,
                ),
                size=mp.Vector3(design_region_x[ii], design_region_y),
            ),
        )
        number_para = Nx * Ny
        # 初始化granting coupler为最终优化结果
        init_para = (np.abs(epses[ii][:, 5]) - 1.44**2) / (1.96**2 - 1.44**2)

        init_para[init_para > 1] = 1
        init_para[init_para < 0] = 0
        design_region.update_design_parameters(init_para)
        geometry.append(
            mp.Block(
                center=design_region.center,
                size=design_region.size,
                material=design_variables,
            )
        )
        wg_gap_geo = mp.Block(
            material=SiN,
            center=mp.Vector3(
                -0.5 * sx
                + input_wvg_length
                + dpml
                + np.sum(design_region_x[0 : (ii + 1)])
                + np.sum(wg_gap[0:ii])
                + 0.5 * wg_gap[ii],
                -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
                0,
            ),
            size=mp.Vector3(wg_gap[ii], h),
        )
        geometry.append(wg_gap_geo)
    # ####################### design region ##########################

    ####################### Sources ##########################
    wvl = 0.785
    fcen = 1 / wvl
    fwidth = 0.2 * fcen
    kpoint = mp.Vector3(1, 0, 0)

    # Oblique planewave source
    k_point = mp.Vector3(fcen * 1).rotate(
        mp.Vector3(z=1), np.radians(-(90 + rot_angle))
    )
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=fwidth),
            center=mp.Vector3(plane_x, plane_y),
            size=mp.Vector3(x=source_x),
            direction=mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.ODD_Z,
            eig_match_freq=True,
        )
    ]
    sources[0].amplitude = 1 / mp.GaussianSource(fcen, fwidth=fwidth).fourier_transform(
        fcen
    )

    ####################### Sources ##########################

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
    )

    # ############### Monitor ###################
    nfreq = 1

    near_region = mp.FluxRegion(
        center=mp.Vector3(0, 0.5 * sy - dpml, 0),
        size=mp.Vector3(sx, 0),
    )

    near_field = sim.add_mode_monitor(
        fcen,
        0,
        1,
        near_region,
    )

    # near2far_field
    n2f_field = sim.add_near2far(
        fcen,
        0,
        1,
        near_region,
    )
    near_dft = sim.add_dft_fields(
        [mp.Ez], fcen, fwidth, nfreq, center=near_region.center, size=near_region.size
    )

    src_region = mp.FluxRegion(
        center=mp.Vector3(
            input_waveguide_geo.center.x + 1, input_waveguide_geo.center.y, 0
        ),
        size=mp.Vector3(0, 3 * h, 0),
    )
    src_field = sim.add_mode_monitor(
        fcen,
        0,
        1,
        src_region,
    )

    out_region = mp.FluxRegion(
        center=mp.Vector3(
            output_waveguide_geo.center.x, output_waveguide_geo.center.y, 0
        ),
        size=mp.Vector3(0, 3 * h, 0),
    )
    out_field = sim.add_mode_monitor(
        fcen,
        0,
        1,
        out_region,
    )

    wg_region = mp.FluxRegion(
        center=mp.Vector3(0, input_waveguide_geo.center.y, 0),
        size=mp.Vector3(sx, 0),
    )
    wg_dft = sim.add_dft_fields(
        [mp.Ez], fcen, fwidth, nfreq, center=wg_region.center, size=wg_region.size
    )
    sim.plot2D(plot_monitors_flag=True)

    # mpt = mp.Vector3(input_waveguide_geo.center.x - 3, input_waveguide_geo.center.y)
    mpt = mp.Vector3(output_waveguide_geo.center.x - 3, input_waveguide_geo.center.y)

    # f = plt.figure(dpi=100)
    # Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)

    sim.run(
        # mp.at_every(1, Animate),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mpt, 1e-5),
    )

    input_mode = sim.get_eigenmode_coefficients(src_field, [1], eig_parity=mp.EVEN_Y)
    output_mode = sim.get_eigenmode_coefficients(out_field, [1], eig_parity=mp.EVEN_Y)
    print("angle: ", rot_angle)

    return abs(input_mode.alpha[0, 0, 1]) ** 2, output_mode.alpha[0, 0, 0] ** 2


angle_resolution = 0.2
port_left_fluxes = []
port_right_fluxes = []
angles = []
for rot_angle in np.linspace(-25, 26, 100):
    port_left_flux, port_right_flux = sweep_rot_angle(rot_angle)
    angles.append(rot_angle)
    port_left_fluxes.append(port_left_flux)
    port_right_fluxes.append(port_right_flux)

np.save(back_path + "port_left_fluxes", port_left_fluxes)
np.save(back_path + "port_right_fluxes", port_right_fluxes)
np.save(back_path + "angles", angles)
