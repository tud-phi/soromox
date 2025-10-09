import sympy as sp

l1, l2 = sp.symbols("l1 l2", nonnegative=True, nonzero=True)
kappa_be1, sigma_sh1, sigma_ax1 = sp.symbols(
    "kappa_be1 sigma_sh1 sigma_ax1", real=True, nonnegative=True, nonzero=True
)
kappa_be2, sigma_sh2, sigma_ax2 = sp.symbols(
    "kappa_be2 sigma_sh2 sigma_ax2", real=True, nonnegative=True, nonzero=True
)

# deactivate shear and axial strains
# sigma_sh1, sigma_sh2 = 0.0, 0.0
# sigma_ax1, sigma_ax2 = 0.0, 0.0

# define the strains for the two segments
q1 = [kappa_be1, sigma_sh1, sigma_ax1]
q2 = [kappa_be2, sigma_sh2, sigma_ax2]

# average the strains
qav = [(l1 * q1[i] + l2 * q2[i]) / (l1 + l2) for i in range(len(q1))]
print("qav =\n", qav)

# apply the forward kinematics on the individual segments
s_be1, c_be1 = sp.sin(kappa_be1 * l1), sp.cos(kappa_be1 * l1)
s_be2, c_be2 = sp.sin(kappa_be2 * l2), sp.cos(kappa_be2 * l2)
chi1 = sp.Matrix(
    [
        [sigma_sh1 * s_be1 / kappa_be1 + sigma_ax1 * (c_be1 - 1) / kappa_be1],
        [sigma_sh1 * (1 - c_be1) / kappa_be1 + sigma_ax1 * s_be1 / kappa_be1],
        [kappa_be1 * l1],
    ]
)
# rotation matrix
R1 = sp.Matrix(
    [
        [c_be1, -s_be1, 0],
        [s_be1, c_be1, 0],
        [0, 0, 1],
    ]
)
chi2_rel = sp.Matrix(
    [
        [sigma_sh2 * s_be2 / kappa_be2 + sigma_ax2 * (c_be2 - 1) / kappa_be2],
        [sigma_sh2 * (1 - c_be2) / kappa_be2 + sigma_ax2 * s_be2 / kappa_be2],
        [kappa_be2 * l2],
    ]
)
chi2 = sp.simplify(chi1 + R1 @ chi2_rel)
# print("chi2 =\n", chi2)
# apply the one-segment inverse kinematic on the distal end of the two segments
px, py, th = chi2[0, 0], chi2[1, 0], chi2[2, 0]
s_th, c_th = sp.sin(th), sp.cos(th)
qfkik = sp.simplify(
    th
    / (2 * (l1 + l2))
    * sp.Matrix(
        [
            [2],
            [py - px * s_th / (c_th - 1)],
            [-px - py * s_th / (c_th - 1)],
        ]
    )
)
print("qfkik =\n", qfkik)
