# On-chip metagrating-based wavefront sensor

## One-dementional metagrating-based wavefront sensor

### 整体优化

#### 远场强度分布优化

1. 正角度优化
    - 生成(0, 25)度 线性增加和线性降低的理想的远场强度分布曲线；
    - 利用MEEP优化metagrating的远场分布为线性增加和线性降低的曲线，优化目标为曲线拟合的MSE最小；（直接考虑优化曲线与理想曲线相乘，优化其乘积最大的方式实际测试不可用）
    - 存在问题：出现部分角度的远场强度分布不满足线性增加和线性降低的曲线，导致优化结果不理想；
2. 负角度优化
   - 考虑实际的光栅耦合方程，只有在光栅周期较大的情况下，输出为正向角度输出；当周期较小时，输出为负角度输出。
   - 正角度拟合会使得metagrating的尺寸过大，因此考虑负角度拟合。
  
    2023.06.16 -> ./negative_angle/inverse_design_negative_angle.py -> 从垂直耦合的metagrating作为初始值，目标耦合效率设定为0.8，迭代次数200 -> 输出：final_eps_negative_2.npy/eval_history_negative_2.npy -> 实际输出耦合效率0.61，在大角度有明显下陷

    2023.06.18 -> ./negative_angle/inverse_design_negative_angle.py -> 从final_eps_negative_2.npy的metagrating作为初始值，**目标耦合效率设定为0.6**，迭代次数200 -> 输出：final_eps_negative_3.npy/eval_history_negative_3.npy -> 在大角度仍然有部分“凹陷”

整体光栅优化时，需要优化的参数很多，且优化目标函数较为复杂，很容易陷入局部最优，从而没有一个良好的优化结果，因此考虑将整个优化目标分割成多个部分进行分段优化。

#### 近场 mode overlap优化 (实际测试不可用)

### 分段优化

#### 分段inverse_design (./segmented_opt)

利用逆向设计分段优化(-25, 0)度的远场分布，0度远场强度最大，-25度远场强度最小。
(./segmented_opt/segmented_base.ipynb)
1. 远场角度每一个角度进行分段(-25, 0)分为26段；
2. 将metagrating分为多个cell unit，每个cell unit 长度 $ 1\mu m $，resolution为20 nm，cell unit包含50个优化单元；
3. 每一个cell对一个远场angle进行优化。
4. 计算出每一个cell unit需要优化的远场目标角度以及散射强度(设定波导出射的flux为一定值)

设置目标函数为 `J1`: 远场在目标角度范围最大化；`J2`: 出射波导中的flux随角度变大逐渐下降

**分段拟合测试：**

`final_eps_seg_test_2`:designed_region_x = 5, 垂直耦合的metagrating, 优化结果在0度时为3.5e-6
`final_eps_seg_test_3`:designed_region_x = 10, 垂直耦合的metagrating, 优化结果在0度时为8e-6
`final_eps_seg_test_4`:designed_region_x = 5, 垂直耦合的metagrating且输出flux为一定值, 优化结果很差
`final_eps_seg_test_5`:designed_region_x = 5, 垂直耦合的metagrating且输出flux为一定值, 优化结果很差

-> 初始化eps为垂直耦合的metagrating，然后再优化其输出flux为一定值.
`final_eps_seg_test_6`:designed_region_x = 5, 以垂直耦合的metagrating为初始值；输出flux有一定的提高，且在0度时的最大。

-> 降低w1值，降低远场目标ob1的占比，提高ob2的占比
`final_eps_seg_test_7`:designed_region_x = 5, 以垂直耦合的metagrating为初始值，降低w1权重为np.power(10, 5)`；输出flux在设定值0.6，且scatter在0度时的最大。**可以满足分段拟合的需求。**

*分段拟合思路：*
1. 以垂直耦合的metagrating为初始值；
2. 设置`w1 = np.power(10, 5)`进行拟合；
3. 迭代次数在100次足够收敛。

1. 以5度进行离散，在0, -5, -10, -15, -20, -25度进行分段拟合垂直耦合，拟合的远场强度的半高全宽角度在5度左右，因此符合实际的需求；
2. 以分段垂直拟合的结果为初始值，逐渐降低输出，再进行分段拟合；
3. 合并分段拟合的结果：
   1. 不同metagrating出射的场之间会相互干扰，为了避免场之间相互干扰，出射的角度为(-20, -15, -10, -5, 0)度;
   2. 合并分段拟合的结果时会缩小远场散射角度的半高全宽，从5度缩小为2.5度左右，因此考虑再2.5um长度内对远场进行分段拟合；远场的角度分段为2.5度，离散角度为(-22.5, -20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, 0)度；

2023.07.24
1. 级联的grating互相之间存在干扰，导致最终的远程分布并不是理想的单独grating远程之和，具体原因还需要进一步分析；
2. 目前返回至上一步中的全局优化，考虑直接进行全局优化，利用多个初始值进行优化，选取其中较好的结果(正角度和负角度)