# 此文件用于总体初始化垂直耦合
# 1.离散的角度(-22.5, -20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0)
# 2. designed_region_x:  3um

import os
import sys

import meep.adjoint as mpa
import numpy as np
from autograd import grad
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy import signal, special
import math
import meep as mp
import nlopt  # need install nlopt

######################## 导入P_angles_scatter ############################
# cur_path = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
# config_path = cur_path.rsplit("/", 1)[0]  # 上一级目录
# sys.path.append(config_path)

# # P_angles_scatter: [-20, -15, -10, -5, 0]
# P_angles_scatter = np.load(config_path + "/P_angles_scatter.npy")
# # plt.plot(P_angles_scatter, "-o")
# # plt.show()
######################## 导入P_angles_scatter ############################

np.random.seed(800)
update_factor = 100  # 迭代次数

for scatter_angle in [-14]:
    ######################## Basic simulation ############################
    # if True compute flux (Gaussian source), if False compute DFT (Continue source)
    compute_flux = True
    # size of input and output waveguide
    w = 0.5
    h = 0.2

    # resolution size grid: 10 nm res: 100
    grid = 0.02
    resolution = 1 / grid

    # thickness of PML
    dpml = 1

    # length parameter
    input_wvg_length = 5
    output_wvg_length = 5
    # !design_region
    design_region_x = 4
    design_region_y = 0.2
    sx = input_wvg_length + design_region_x + output_wvg_length + 2 * dpml

    # height parameter
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

    cell = mp.Vector3(sx, sy, 0)

    # Material setting
    Si = mp.Medium(index=3.45)
    SiO2 = mp.Medium(index=1.44)
    SiN = mp.Medium(index=1.96)

    # geometry
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
            -0.5 * sx + 0.5 * input_wvg_length + dpml,
            -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
        ),
        size=mp.Vector3(input_wvg_length, h),
    )
    output_waveguide_geo = mp.Block(
        material=SiN,
        center=mp.Vector3(
            0.5 * sx - 0.5 * output_wvg_length - dpml,
            -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
        ),
        size=mp.Vector3(output_wvg_length, h),
    )

    geometry = [
        Substrate_geo,
        BOX_geo,
        TOX_geo,
        input_waveguide_geo,
        output_waveguide_geo,
    ]

    design_region_resolution_x = int(resolution)  # res=50,  20nm resolution
    design_region_resolution_y = 5  # res=5, 200nm resolution
    Nx = int(design_region_resolution_x * design_region_x)
    Ny = int(design_region_resolution_y * design_region_y)
    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_DEFAULT"
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(
                -0.5 * sx + input_wvg_length + 0.5 * design_region_x + dpml,
                -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
                0,
            ),
            size=mp.Vector3(design_region_x, design_region_y),
        ),
    )
    geometry.append(
        mp.Block(
            center=design_region.center,
            size=design_region.size,
            material=design_variables,
        )
    )

    # Sources
    wvl = 0.785
    fcen = 1 / wvl
    fwidth = 0.2 * fcen
    kpoint = mp.Vector3(1, 0, 0)

    sources = [
        mp.EigenModeSource(
            mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=input_waveguide_geo.center,
            size=mp.Vector3(0, 3 * h),
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z,
            eig_match_freq=True,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
        )
    ]
    # set nomrlized source: amplitude is 1 -> flux is 1
    sources[0].amplitude = 1 / mp.GaussianSource(fcen, fwidth=fwidth).fourier_transform(
        fcen
    )

    # Simulation
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
        force_all_components=True,
    )

    ######################## Basic simulation ############################

    ######################## Opt settings ############################
    # # 输出波导的flux
    # mode = 1
    # output_field = mp.Volume(
    #     center=output_waveguide_geo.center, size=mp.Vector3(0, 3 * h, 0)
    # )
    # out = mpa.EigenmodeCoefficient(sim, output_field, mode)
    # idea_output = 0.6

    # near field region
    NearRegion = [
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0.5 * sy - dpml - 0.5 * near_field_height, 0),
            size=mp.Vector3(sx - 2 * dpml, 0),
            weight=+1,
        )
    ]

    ff_distance = 1e6
    fx = np.tan(np.radians(scatter_angle)) * ff_distance

    ff_points = [mp.Vector3(fx, ff_distance, 0)]
    near_Ez = mpa.Near2FarFields(sim, NearRegion, ff_points)
    # ob_list
    # point |E|^2: (3-8 e-6)
    ob_list = [near_Ez]

    # def J1(angle_point, out):
    #     # ob1为角度最大的目标函数，单目标优化的目标值在0.3左右
    #     out = np.squeeze(out)
    #     # 权重
    #     w1 = np.power(10, 5)
    #     ob1 = (npa.abs(angle_point[0, 0, 2]) ** 2) * w1
    #     # ob2为输出波导的目标函数，目标优化值在-0.1左右
    #     w2 = 1
    #     ob2 = -npa.abs((npa.abs(out) ** 2 - idea_output) * w2)
    #     ob = ob1 + ob2
    #     return ob
    def J1(FF):
        return npa.mean(npa.abs(FF[0, 0, 2]) ** 2)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J1],
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=[fcen],
        decay_by=1e-5,
    )

    # define the initial design and confirm the optimization problem
    number_para = Nx * Ny

    # 初始化granting coupler为垂直耦合的eps
    # 将 [1.44, 1.96] 的 eps 映射到 [0, 1] 之间
    # init_para = (
    #     np.abs(np.load("final_eps_seg_test_6.npy")[0:number_para, 5]) - 1.44**2
    # ) / (1.96**2 - 1.44**2)
    init_para = np.random.random(number_para)
    init_para[init_para < 0] = 0
    init_para[init_para > 1] = 1

    # 随机初始化
    # init_para = np.random.random(number_para)
    opt.update_design([init_para[0:number_para]])

    evaluation_history = []
    cur_iter = [0]

    def f(v, gradient):
        print("Current iteration: {}".format(cur_iter[0] + 1))

        f0, dJ_du = opt([v])  # compute objective and gradient

        # plt.figure()
        # ax = plt.gca()
        # opt.plot2D(
        #     False,
        #     ax=ax,
        #     plot_sources_flag=False,
        #     plot_monitors_flag=False,
        #     plot_boundaries_flag=False,
        # )
        # circ = Circle((2, 2), minimum_length / 2)
        # ax.add_patch(circ)
        # ax.axis("off")
        # plt.show()

        if gradient.size > 0:
            gradient[:] = np.squeeze(dJ_du)

        evaluation_history.append(np.real(f0))

        print("Objective function: {}".format(np.real(f0)))
        print("scatter_angle: {}".format(scatter_angle))
        cur_iter[0] = cur_iter[0] + 1

        return np.real(f0)

    algorithm = nlopt.LD_MMA
    n = Nx * Ny  # number of parameters

    # Initial guess
    # x = np.random.random((n,)) * 0.5
    x = init_para[0:number_para]

    # lower and upper bounds
    lb = np.zeros((Nx * Ny,))
    ub = np.ones((Nx * Ny,))

    ftol = 1e-5
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    # !set objective to max or min
    solver.set_max_objective(lambda a, g: f(a, g))
    solver.set_maxeval(update_factor)
    # solver.set_ftol_rel(ftol)
    x[:] = solver.optimize(x)

    ######################## Opt settings ############################

    # save evalution_history and eps
    np.save(f"eval_history_seg_init_{scatter_angle}.npy", evaluation_history)

    eps = opt.sim.get_array(
        component=mp.Dielectric,
        frequency=fcen,
        center=mp.Vector3(
            -0.5 * sx + dpml + input_wvg_length + 0.5 * design_region_x,
            -0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * h + dpml,
            0,
        ),
        size=mp.Vector3(design_region_x, h, 0),
    )
    np.save(f"final_eps_seg_init_{scatter_angle}.npy", eps)
