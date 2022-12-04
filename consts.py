from distributions import UniformDist

default_random_seed = 42
x_dist_default = UniformDist
# y0_dist_param_default = {'a0': 0.0, 'b0': 2.0, 'c0': 100.0, 'sigma_01': 1, 'sigma_00': 1}
# y1_dist_param_default = {'a1': 0.0, 'b1': 4.0, 'c1': 100.0, 'sigma_11': 1, 'sigma_10': 1}
y0_dist_param_default = {'a0': 0.0, 'b0': 2.0, 'c0': 10.0, 'sigma_01': 1, 'sigma_00': 1}
y1_dist_param_default = {'a1': 0.0, 'b1': 4.0, 'c1': 10.0, 'sigma_11': 1, 'sigma_10': 1}
treatment_prob_default = 0.5
omega_default = 1.0
beta_d_default = [-2.0, -2.0, 1.0]
population_size_default = 10000