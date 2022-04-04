import numpy as np
import create
import generate
import recall



# Params
n = 100
p = 0.5
r = 0.1
m = 10

mu_gaus = 3.0*np.sqrt(0.1)
sigma_gaus = np.sqrt(0.1)

xs, ss = generate.memories_gaussian(n, p, m, mu_gaus, sigma_gaus)
ss[0] = 1.0
w_gaus = generate.weight_matrix(xs, ss)
w_pseudo = generate.pseudo_matrix(xs, ss)
x_t = generate.noise_exact(xs[0], r)

update_x = create.update_x(beta=1.0)
update_s = create.update_s_known()

x_samples, s_samples = recall.gibbs(w_gaus, m, x_t, 1.0, update_x, update_s, 100)

x_recall = np.median(x_samples[10:], 0)
s_recall = np.mean(s_samples[10:])
print(s_recall)

