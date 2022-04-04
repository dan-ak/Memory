import numpy as np
import scipy.stats as stats



def update_x_marginal(mu_s, sigma_s, c=1.0, beta=1000.0):
    e_s2 = mu_s * mu_s + sigma_s * sigma_s

    def update(W, x_old, x_t, s, k, M, r=0.10):
        x = np.copy(x_old)

        H = np.log(r / (1 - r)) + x_t[k] * 2 * np.log((1 - r) / r)
        H += 16 * mu_s * c * (np.dot(W[k,], x) - W[k, k] * x[k]) / ((M - 1) * e_s2 + sigma_s * sigma_s)
        H += - 8 * mu_s * c * (np.sum(W[k,]) - W[k, k]) / ((M - 1) * e_s2 + sigma_s * sigma_s)

        x[k] = 1 * (H > 0)

        if (beta < 1000):
            sigma = 1 / (1 + np.exp(-beta * H))
            x[k] = np.random.binomial(1, sigma)

        return x

    return update

def update_x_marginal_exp(b, c=1.0, beta=1000.0):
    e_s2 = 2*b**2

    def update(W, x_old, x_t, s, k, M, r=0.10):
        x = np.copy(x_old)
        N = len(x)

        H1 = np.log(r / (1 - r)) + x_t[k] * 2 * np.log((1 - r) / r)

        H2 = 0
        H3 = 0

        var = (M - 1) * e_s2 * 0.5 ** 4
        for j in range(N):
            if k != j :
                H2 += -16 * c / b * W[k, j] * (1 - 0.5) * (x[j] - 0.5)

                H2 += 16 * c / b * W[k, j] * (0 - 0.5) * (x[j] - 0.5)

                mu1 = 16 * (W[k, j] * (1 - 0.5) * (x[j] - 0.5) - var / b)
                int1 = stats.norm.cdf(mu1 / np.sqrt(16*var))

                mu0 = 16 * (W[k, j] * (0 - 0.5) * (x[j] - 0.5) - var / b)
                int0 = stats.norm.cdf(mu0 / np.sqrt(16 * var))

                H3 += c * (np.log(int1) - np.log(int0))

        H = H1 + H2 + H3

        x[k] = 1 * (H > 0)

        if (beta < 1000):
            sigma = 1 / (1 + np.exp(-beta * H))
            x[k] = np.random.binomial(1, sigma)

        return x

    return update


def update_x_gaus_exact(mu, sigma_s, c=1.0, beta=1000.0):
    e_s2 = mu**2 + sigma_s**2

    def update(W, x_old, x_t, s, k, M, r=0.10):
        x = np.copy(x_old)
        N = len(x)
        x1 = np.copy(x_old); x1[k] = 1
        x0 = np.copy(x_old); x0[k] = 0

        dW1 = np.triu(np.outer(x1-0.5, x1-0.5), k=1).flatten()
        dW0 = np.triu(np.outer(x0-0.5, x0-0.5), k=1).flatten()

        s1 = np.dot(dW1, W.flatten())
        s0 = np.dot(dW0, W.flatten())

        var = (M-1)*e_s2*0.5**4
        s2 = N*(N-1)*0.5**4

        ####
        H1 = (s1 + mu * var / sigma_s**2)**2 / (2 * var * (s2 + var / sigma_s**2))
        H0 = (s0 + mu * var / sigma_s**2)**2 / (2 * var * (s2 + var / sigma_s**2))

        H = np.log(r / (1 - r)) + x_t[k] * 2 * np.log((1 - r) / r)
        H += c * (H1 - H0)

        x[k] = 1 * (H > 0)

        if (beta < 1000):
            sigma = 1 / (1 + np.exp(-beta * H))
            x[k] = np.random.binomial(1, sigma)

        return x

    return update

def update_x_exp_exact(b, c=1.0, beta=1000.0):
    e_s2 = 2*b**2

    def update(W, x_old, x_t, s, k, M, r=0.10):
        x = np.copy(x_old)
        N = len(x)
        x1 = np.copy(x_old); x1[k] = 1
        x0 = np.copy(x_old); x0[k] = 0

        dW1 = np.triu(np.outer(x1-0.5, x1-0.5), k=1).flatten()
        dW0 = np.triu(np.outer(x0-0.5, x0-0.5), k=1).flatten()

        s1 = np.dot(dW1, W.flatten())
        s0 = np.dot(dW0, W.flatten())
        var = (M-1)*e_s2*0.5**4
        s2 = N*(N-1)*0.5**4

        mu1 = (s1 - var / b) / s2
        var1 = var / s2
        #int1 = stats.norm.cdf(mu1 / np.sqrt(var1))
        H1 = (s1 - var / b) ** 2 / (2 * var * s2) #+ np.log(int1)

        mu0 = (s0 - var / b) / s2
        var0 = var / s2
        #int0 = stats.norm.cdf(mu0 / np.sqrt(var0))
        H0 = (s0 - var / b) ** 2 / (2 * var * s2) #+ np.log(int0)

        H = np.log(r / (1 - r)) + x_t[k] * 2 * np.log((1 - r) / r)
        H += c * (H1 - H0)

        x[k] = 1 * (H > 0)

        if (beta < 1000):
            sigma = 1 / (1 + np.exp(-beta * H))
            x[k] = np.random.binomial(1, sigma)

        return x, H, c * (H1 - H0)

    return update

def update_x_hop(c_func=lambda snr: 1, e_s2=1.0, beta=1000.0):
    def update(W, x_old, x_t, s, k, M, r=0.10):

        snr = s * s / ((M - 1) * e_s2)
        c = c_func(snr)

        x = np.copy(x_old)

        H = (np.dot(W[k,], x) - W[k, k] * x[k])

        x[k] = 1 * (H > 0)

        if (beta < 1000):
            sigma = 1 / (1 + np.exp(- beta * H))
            x[k] = np.random.binomial(1, sigma)

        return x

    return update

def update_x(c_func=lambda snr: 1, e_s2=1.0, beta=1000.0):
    def update(W, x_old, x_t, s, k, M, r=0.10):

        snr = s * s / ((M - 1) * e_s2)
        c = c_func(snr)

        x = np.copy(x_old)

        H = np.log(r / (1 - r)) + x_t[k] * 2 * np.log((1 - r) / r)
        H += 16 * c * s / ((M - 1) * e_s2) * (np.dot(W[k,], x) - W[k, k] * x[k])
        H += - 8 * c * s / ((M - 1) * e_s2) * (np.sum(W[k,]) - W[k, k])

        x[k] = 1 * (H > 0)

        if (beta < 1000):
            sigma = 1 / (1 + np.exp(- beta * H))
            x[k] = np.random.binomial(1, sigma)

        return x

    return update

def update_x_flip(c_func=lambda snr: 1, e_s2=1.0, beta=1000.0):
    def update(W, x_old, x_t, s, k1, k2, M, r=0.10):

        snr = s * s / ((M - 1) * e_s2)
        c = c_func(snr)

        x = np.copy(x_old)

        H = 2 * np.log((1 - r) / r) * (x[k1]*x_t[k1] + x[k2]*x_t[k2] - x[k2]*x_t[k1] - x[k1]*x_t[k2])

        term_sum = 0

        for i in range(len(x)):
            if i != k1 and i != k2:
                term_sum += 4 * (x[k1]*x[i]*W[k1, i] + x[k2]*x[i]*W[k2, i] - x[k1]*x[i]*W[k2, i] - x[k2]*x[i]*W[k1, i])
                term_sum += 2 * (W[k1, i]*(x[k1]+x[i]) + W[k2, i]*(x[k2]+x[i]) - W[k1, i]*(x[k2]+x[i]) - W[k2, i]*(x[k1]+x[i]))


        H += 16 * c * s / ((M - 1) * e_s2) * term_sum


        if (beta < 1000):
            sigma = 1 / (1 + np.exp(- beta * H))
            flip = np.random.binomial(1, 1 - sigma)
            if flip:
                temp = x[k1]
                x[k1] = x[k2]
                x[k2] = temp
        else:
            if (H < 0):
                temp = x[k1]
                x[k1] = x[k2]
                x[k2] = temp

        return x

    return update


################################
#######Update strength##########
################################

def update_s_known():

    def update_s(W, x, M, s, p = 0.5, l = 1.0):
        return s

    return update_s


def update_s_ff(mu_p=0, sigma=10000, s_factor=1.0, beta=1000):
    def update_func(W, x, M, s_old):

        assert len(W.shape) == 1, 'W_ff has more than one dimension'

        N = len(x)
        total = 0
        var_p = sigma**2

        for i in range(N):
            total += 4 * W[i] * (-1) ** (1 + x[i])

        count = N
        mu_W = total / count
        var_W = (M - 1) / count * s_factor

        mean = mu_W
        var = var_W

        if sigma < 1000:
            mean = (var_p * mu_W + var_W * mu_p) / (var_p + var_W)
            var = 1/(1/var_p + 1/var_W)

        if beta == 1000:
            return mean
        else:
            return np.random.normal(mean, np.sqrt(var))

    return update_func

def update_s_exp(b=1, s_factor=1.0, beta=1000):
    def update_func(W, x, M, s_old):

        e_s2 = 2*b**2
        N = len(x)

        total = 4*np.sum(np.triu(W*(-1)**(x[:, None]+x), 1))

        count = (N * (N - 1)) / 2
        mu_W = total / count
        var_W = (M - 1) * e_s2 / count / s_factor

        mean = mu_W - var_W*b
        var = var_W

        if beta == 1000:
            return np.max([0.00001, mean])
        else:
            return stats.truncnorm(-mean/np.sqrt(var), np.inf, mean, np.sqrt(var)).rvs()

    return update_func

def update_s(mu_p=0, sigma=10000, s_factor=1.0, beta=1000):
    def update_func(W, x, M, s_old):

        N = len(x)
        total = 0
        var_p = sigma**2
        e_s2 = mu_p**2 + sigma**2

        #for i in range(N):
        #    for j in range(i):
        #        total += 4 * W[i, j] * (-1) ** (x[i] + x[j])

        total = 4*np.sum(np.triu(W*(-1)**(x[:, None]+x), 1))

        count = (N * (N - 1)) / 2
        mu_W = total / count
        var_W = (M - 1) * e_s2 / count / s_factor

        mean = mu_W
        var = var_W

        if sigma < 1000:
            mean = (var_p * mu_W + var_W * mu_p) / (var_p + var_W)
            var = 1/(1/var_p + 1/var_W)

        if beta == 1000:
            return mean
        else:
            return np.random.normal(mean, np.sqrt(var))

    return update_func

def update_s_frac2(mu_p=0, sigma=10000, s_factor=1.0, beta=1000, frac=1):
    def update_func(W, x, M, s_old):

        N = len(x)
        total = 0
        var_p = sigma ** 2
        e_s2 = mu_p ** 2 + sigma ** 2

        # for i in range(N):
        #    for j in range(i):
        #        total += 4 * W[i, j] * (-1) ** (x[i] + x[j])

        total = 4 * np.sum(np.triu(W * (-1) ** (x[:, None] + x), 1))

        count = (N * (N - 1))*frac / 2
        mu_W = total / (count)
        var_W = (M - 1) * e_s2 / (count) / s_factor

        mean = mu_W
        var = var_W

        if sigma < 1000:
            mean = (var_p * mu_W + var_W * mu_p) / (var_p + var_W)
            var = 1/(1/var_p + 1/var_W)

        if beta == 1000:
            return mean
        else:
            return np.random.normal(mean, np.sqrt(var))

    return update_func

def update_s_frac(mu_p=0, sigma=10000, s_factor=1.0, beta=1000, frac=1):
    def update_func(W, x, M, s_old):

        N = len(x)
        total = 0
        count = (N * (N - 1)) / 2

        var_p = sigma**2
        e_s2 = mu_p**2 + sigma**2

        w_on = np.zeros(np.int(count))
        w_on[:np.int(count*frac)] = 1
        np.random.shuffle(w_on)

        mask = np.zeros((N, N))
        mask[np.triu_indices(N, 1)] = w_on

        W_f = mask * W

        #for i in range(N):
        #    for j in range(i):
        #        total += 4 * W_f[i, j] * (-1) ** (x[i] + x[j])

        #total = 4*np.sum(np.triu(W*(-1)**(x[:, None]+x), 1))
        total = 4*np.sum(W_f*(-1)**(x[:, None]+x))

        mu_W = total / (count*frac)
        var_W = (M - 1) * e_s2 / (count*frac) / s_factor

        mean = mu_W
        var = var_W

        if sigma < 1000:
            mean = (var_p * mu_W + var_W * mu_p) / (var_p + var_W)
            var = 1/(1/var_p + 1/var_W)

        if beta == 1000:
            return mean
        else:
            return np.random.normal(mean, np.sqrt(var))

    return update_func


def update_s_c(mu_p=0, sigma=10000, s_factor=1.0, c_func=lambda snr: c, beta=1000):
    def update_func(W, x, M, s_old):

        N = len(x)
        total = 0
        var_p = sigma ** 2
        e_s2 = mu_p ** 2 + sigma ** 2

        # for i in range(N):
        #    for j in range(i):
        #        total += 4 * W[i, j] * (-1) ** (x[i] + x[j])

        total = 4 * np.sum(np.triu(W * (-1) ** (x[:, None] + x), 1))

        count = (N * (N - 1)) / 2
        mu_W = total / count
        var_W = (M - 1) * e_s2 / count * s_factor

        mean = mu_W
        var = var_W

        if sigma < 1000:
            mean = (var_p * mu_W + var_W * mu_p) / (var_p + var_W)
            var = 1 / (1 / var_p + 1 / var_W)

        if beta == 1000:
            return mean
        else:

            k = np.sqrt(10)

            q = lambda s: stats.norm.pdf(s, loc=mean, scale=np.sqrt(10.0 * var))

            def p(s):
                return stats.norm.pdf(s, loc=mu_W, scale=np.sqrt(var_W / c_func(s))) * \
                       stats.norm.pdf(s, loc=mu_p, scale=sigma)

            ss = np.linspace(-1, 4, 101)

            k = 1.1 * max(p(ss) / q(ss))

            for i in range(10000):
                z = np.random.normal(mean, np.sqrt(10.0 * var))
                u = np.random.uniform(0, k * q(z))

                if u <= p(z):
                    return z

            return mean

    return update_func


###OLD####
def update_s_uni(p=0.5, beta=1000):
    def update_s(W_old, x, M, s_old):
        W = W_old.copy()
        np.fill_diagonal(W, 0)
        N = len(x)
        E_s2 = 1.0

        var_s = (4 * (M - 1) * E_s2) / (N * N - 1)
        mu_s = np.dot((x - p).T, np.dot(W, x - p)) / (4 * (M - 1) * E_s2 * p * p * (1 - p) * (1 - p))
        mu_s = mu_s * var_s

        if (beta == 1):
            s = np.random.normal(mu_s, np.sqrt(var_s))
        else:
            s = mu_s

        return np.amax([0.0000001, s])

    return update_s


def update_s_gaus(mu, sigma, p=0.5, beta=1000):
    def update_s(W_old, x, M, s_old):
        W = W_old.copy()
        np.fill_diagonal(W, 0)
        N = len(x)
        E_s2 = mu * mu + sigma * sigma

        var_s = 1 / (1 / (sigma * sigma) + (N * N - 1) / ((4 * (M - 1) * E_s2)))

        mu_s = np.dot((x - p).T, np.dot(W, x - p)) / (4 * (M - 1) * E_s2 * p * p * (1 - p) * (1 - p)) + mu / (
                    sigma * sigma)
        mu_s = mu_s * var_s

        if (beta == 1):
            s = np.random.normal(mu_s, np.sqrt(var_s))
        else:
            s = mu_s

        return np.amax([0.0000001, s])

    return update_s