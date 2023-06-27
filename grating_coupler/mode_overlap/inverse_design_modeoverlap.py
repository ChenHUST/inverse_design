# 2023-06-15
# 1 修改目标函数：
#   1.1 以 mode overlap 为目标函数，设置为目标函数最大
# 2. 以random parameter开始
# 3. 迭代次数设置 200
# 4. 未添加结构对称
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

######################## Basic simulation ############################
np.random.seed(2000)
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

# length parameter
input_wvg_length = 5
output_wvg_length = 5
# design_region
design_region_x = 15
design_region_y = 0.2
sx = input_wvg_length + design_region_x + output_wvg_length + 2 * dpml

# height parameter
Substrate_thickness = 0.5
BOX_thickness = 2
TOX_thickness = 0.7
near_field_height = 1
sy = BOX_thickness + TOX_thickness + Substrate_thickness + near_field_height + 2 * dpml

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
    center=mp.Vector3(y=-0.5 * sy + Substrate_thickness + 0.5 * BOX_thickness + dpml),
    size=mp.Vector3(sx, BOX_thickness),
)
TOX_geo = mp.Block(
    material=SiO2,
    center=mp.Vector3(
        y=-0.5 * sy + Substrate_thickness + BOX_thickness + 0.5 * TOX_thickness + dpml
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

# design region
minimum_length = 0.05  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)

design_region_resolution_x = int(resolution)
design_region_resolution_y = 5  # 200nm resolution
Nx = int(design_region_resolution_x * design_region_x)
Ny = int(design_region_resolution_y * design_region_y)
# 结构对称
# design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_MEAN")
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_DEFAULT")
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
)

######################## Basic simulation ############################

######################## Opt settings ############################

# near field region
NearRegion = mp.Volume(
    center=mp.Vector3(0, 0.5 * sy - dpml - 0.5 * near_field_height, 0),
    size=mp.Vector3(sx - 2 * dpml, 0),
)

# ob_list
ob_list = [mpa.FourierFields(sim, NearRegion, mp.Ez)]

# !目标dft电场分布
norm_near_field = np.load("near_field.npy")


def J4(FF):
    return npa.abs(npa.sum(npa.squeeze(FF) * npa.conj(norm_near_field)))


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J4],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=[fcen],
    decay_by=1e-5,
)

# define the initial design and confirm the optimization problem
number_para = Nx * Ny

# # 初始化grating coupler为周期性结构
# SiN_mesh = np.ones(
#     shape=[
#         18,
#     ]
# )
# etch_mesh = np.zeros(
#     shape=[
#         17,
#     ]
# )
# single_period = np.concatenate((SiN_mesh, etch_mesh), axis=0)
# init_para = np.tile(single_period, 22)

# 初始化granting coupler为上一次优化的结果
# init_para = (np.abs(np.load("final_eps_6.npy")[0:number_para, 5]) - 1.44**2) / (
#     1.96**2 - 1.44**2
# )

# init_para[init_para < 0] = 0
# init_para[init_para > 1] = 1
# opt.update_design([init_para[0:number_para]])

# 随机初始化
init_para = np.random.random(number_para)
opt.update_design([init_para[0:number_para]])

evaluation_history = []
cur_iter = [0]


# def mapping(x):
#     projected_field = (npa.flipud(x) + x) / 2  # left-right symmetry
#     # interpolate to actual materials
#     return projected_field.flatten()


def f(v, gradient):
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0, dJ_du = opt([v])  # compute objective and gradient

    if gradient.size > 0:
        gradient[:] = np.squeeze(dJ_du)

    evaluation_history.append(np.real(f0))

    print("Objective function: {}".format(np.real(f0)))

    cur_iter[0] = cur_iter[0] + 1

    return np.real(f0)


# def f(v, gradient):
#     print("Current iteration: {}".format(cur_iter[0] + 1))

#     f0, dJ_du = opt([mapping(v)])  # compute objective and gradient

#     if gradient.size > 0:
#         gradient[:] = np.squeeze(dJ_du)

#     evaluation_history.append(np.real(f0))

#     print("Objective function: {}".format(np.real(f0)))

#     cur_iter[0] = cur_iter[0] + 1

#     return np.real(f0)


algorithm = nlopt.LD_MMA
n = Nx * Ny  # number of parameters

# Initial guess
# x = np.random.random((n,)) * 0.5
x = init_para[0:number_para]

# lower and upper bounds
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))
# cur_beta = 4
# beta_scale = 2
# num_betas = 6
update_factor = 200
ftol = 1e-5
solver = nlopt.opt(algorithm, n)
solver.set_lower_bounds(lb)
solver.set_upper_bounds(ub)
solver.set_max_objective(lambda a, g: f(a, g))
solver.set_maxeval(update_factor)
# solver.set_ftol_rel(ftol)
x[:] = solver.optimize(x)

######################## Opt settings ############################

# save evalution_history and eps
np.save("eval_history_modeoverlap_2.npy", evaluation_history)

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
np.save("final_eps_modeoverlap_2.npy", eps)
