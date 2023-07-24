import meep as mp
import math
import matplotlib.pyplot as plt
import numpy as np
import os


def sweep_rot_angle(rot_angle=25):
    # if True compute flux (Gaussian source), if False compute DFT (Continue source)
    compute_flux = True
    # size of input and output waveguide
    w = 0.5
    h = 0.2

    # resolution size
    grid = 0.02
    resolution = 1 / grid

    # thickness of PML
    dpml = 1

    ##################### length parameter ################
    input_wvg_length = 10
    output_wvg_length = 5

    # period = 0.707  # period of grating coupler
    period = 0  # period of grating coupler
    duty_cycle = 0  # duty cycle of grating coupler
    period_2 = 0.707  # period of grating coupler
    duty_cycle_2 = 0.53  # duty cycle of grating coupler
    number_period = 5  # number of period of grating coupler
    sx = (
        input_wvg_length
        + period * number_period
        + period_2 * number_period
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
        -0.5 * sx + input_wvg_length + 0.5 * (period_2 + period) * number_period + dpml
    )
    source_x = 7

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

    grating_coupler_geo = []
    for ii in range(1, number_period + 1):
        grating_coupler_geo.append(
            mp.Block(
                material=SiN,
                center=mp.Vector3(
                    -0.5 * sx
                    + input_wvg_length
                    + ii * period
                    - duty_cycle * period / 2
                    + dpml,
                    -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
                ),
                size=mp.Vector3(duty_cycle * period, h),
            )
        )

    grating_coupler_geo_2 = []
    for ii in range(1, number_period + 1):
        grating_coupler_geo_2.append(
            mp.Block(
                material=SiN,
                center=mp.Vector3(
                    -0.5 * sx
                    + input_wvg_length
                    + number_period * period
                    + ii * period_2
                    - duty_cycle_2 * period_2 / 2
                    + dpml,
                    -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
                ),
                size=mp.Vector3(duty_cycle_2 * period_2, h),
            )
        )

    geometry = [
        Substrate_geo,
        BOX_geo,
        TOX_geo,
        input_waveguide_geo,
        output_waveguide_geo,
        *grating_coupler_geo,
        *grating_coupler_geo_2,
    ]

    ####################### geometry ##########################

    ####################### Sources ##########################
    wvl = 0.785
    fcen = 1 / wvl
    fwidth = 0.2 * fcen
    kpoint = mp.Vector3(1, 0, 0)

    # sources = [
    #     mp.EigenModeSource(
    #         mp.GaussianSource(frequency=fcen, fwidth=fwidth),
    #         center=input_waveguide_geo.center,
    #         size=mp.Vector3(0, 3 * h),
    #         eig_band=1,
    #         eig_parity=mp.EVEN_Y + mp.ODD_Z,
    #         eig_match_freq=True,
    #         direction=mp.NO_DIRECTION,
    #         eig_kpoint=kpoint,
    #     )
    # ]
    # # set nomrlized source: amplitude is 1 -> flux is 1

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

    # src_region
    src_region = mp.FluxRegion(
        center=input_waveguide_geo.center - mp.Vector3(input_wvg_length / 3, 0, 0),
        size=mp.Vector3(0, 5 * h, 0),
    )
    src_flux = sim.add_mode_monitor(fcen, 0, 1, src_region)

    src_dft = sim.add_dft_fields(
        [mp.Ez], fcen, fwidth, nfreq, center=src_region.center, size=src_region.size
    )

    # near_field_region
    near_region = mp.FluxRegion(
        center=mp.Vector3(plane_x, plane_y - 0.1, 0),
        size=mp.Vector3(source_x + 1, 0),
    )

    near_field = sim.add_mode_monitor(
        fcen,
        fwidth,
        nfreq,
        near_region,
    )

    near_dft = sim.add_dft_fields(
        [mp.Ez], fcen, fwidth, nfreq, center=near_region.center, size=near_region.size
    )

    # near2far_field
    n2f_field = sim.add_near2far(
        fcen,
        0,
        nfreq,
        near_region,
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

    # run simulation
    # monitor point
    mpt = mp.Vector3(0.5 * sx - dpml - 1, output_waveguide_geo.center.y)

    f = plt.figure(dpi=100)
    Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)

    sim.run(
        mp.at_every(1, Animate),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mpt, 1e-5),
    )

    input_mode = sim.get_eigenmode_coefficients(src_flux, [1], eig_parity=mp.EVEN_Y)
    output_mode = sim.get_eigenmode_coefficients(out_field, [1], eig_parity=mp.EVEN_Y)
    print("angle: ", rot_angle)

    return abs(input_mode.alpha[0, 0, 1]) ** 2, output_mode.alpha[0, 0, 0] ** 2


angle_resolution = 1
port_left_fluxes = []
port_right_fluxes = []
angles = []

for rot_angle in range(-45, 45 + 1, angle_resolution):
    port_left_flux, port_right_flux = sweep_rot_angle(rot_angle)
    angles.append(rot_angle)
    port_left_fluxes.append(port_left_flux)
    port_right_fluxes.append(port_right_flux)

np.save("port_left_fluxes", port_left_fluxes)
np.save("port_right_fluxes", port_right_fluxes)
np.save("angles", angles)
