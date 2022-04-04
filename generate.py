import numpy as np


class DataTrialsGaus:
    def __init__(self, trials, N, p, m, r, mu, sigma):
        self.trials = trials
        self.N = N
        self.p = p
        self.m = m
        self.xs = np.zeros([trials, m, N])
        self.xts = np.zeros([trials, N])
        self.ss = np.zeros([trials, m])
        self.W = np.zeros([trials, N, N])
        self.Wp = np.zeros([trials, N, N])
        self.mu = mu
        self.sigma = sigma
        self.S2 = mu**2 + sigma**2

        for i in range(trials):

            xs, ss = memories_gaussian(N, p, m, mu, sigma)
            ss[0] = 0.0
            self.W[i] = weight_matrix(xs, ss, 0.5)
            self.Wp[i] = pseudo_matrix(xs, ss, 0.5)
            self.xts[i] = noise_exact(xs[0], r)
            self.xs[i] = xs
            self.ss[i] = ss

    def with_snr(self, t, snr):
        s0 = np.sqrt(snr * (self.m-1) * self.S2)
        W = np.copy(self.W[t]) + s0 * np.outer(self.xs[t][0]-0.5, self.xs[t][0]-0.5)
        Wp = np.copy(self.Wp[t]) + s0 * np.outer(self.xs[t][0]-0.5, self.xs[t][0]-0.5)

        return W, Wp, s0, self.xs[t][0], self.xts[t]

class DataTrialsExp:
    def __init__(self, trials, N, p, m, r, beta):
        self.trials = trials
        self.N = N
        self.p = p
        self.m = m
        self.xs = np.zeros([trials, m, N])
        self.xts = np.zeros([trials, N])
        self.ss = np.zeros([trials, m])
        self.W = np.zeros([trials, N, N])
        self.Wp = np.zeros([trials, N, N])
        self.beta = beta
        self.S2 = 2*beta**2

        for i in range(trials):

            xs, ss = memories_exponential(N, p, m, beta)
            ss[0] = 0.0
            self.W[i] = weight_matrix(xs, ss, 0.5)
            self.Wp[i] = pseudo_matrix(xs, ss, 0.5)
            self.xts[i] = noise_exact(xs[0], r)
            self.xs[i] = xs
            self.ss[i] = ss

    def with_snr(self, t, snr):
        s0 = np.sqrt(snr * (self.m-1) * self.S2)
        W = np.copy(self.W[t]) + s0 * np.outer(self.xs[t][0]-0.5, self.xs[t][0]-0.5)
        Wp = np.copy(self.Wp[t]) + s0 * np.outer(self.xs[t][0]-0.5, self.xs[t][0]-0.5)

        return W, Wp, s0, self.xs[t][0], self.xts[t]

def random_x(n, p):
    x = np.random.random(size=n)
    return 1.0*(x < p)


def noise(x, r):
    x_t = np.copy(x)
    N = x_t.shape[0]
    flip = 1.0*(np.random.random(size=N) < r)
    np.random.shuffle(flip)
    return (x_t + flip) % 2

def noise_exact(x, r):
    x_t = np.copy(x)
    N = x_t.shape[0]
    N_f = int(N * r)
    flip = np.array([0] * (N-N_f) + [1]*N_f)
    np.random.shuffle(flip)
    return (x_t + flip) % 2


def gaussian_s(mu, sigma):
    #s = np.max([0.0, np.random.normal(mu, sigma)])
    s = np.random.normal(mu, sigma)
    return s


def memories_gaussian(n, p, m, mu, sigma):

    xs = []
    ss = []

    for i_m in range(m):
        x = random_x(n, p)
        s = gaussian_s(mu, sigma)
        xs.append(x)
        ss.append(s)

    return np.asarray(xs), np.asarray(ss)

def memories_exponential(n, p, m, beta):

    xs = []
    ss = []

    for i_m in range(m):
        x = random_x(n, p)
        s = np.random.exponential(beta)
        xs.append(x)
        ss.append(s)

    return np.asarray(xs), np.asarray(ss)

def fam_weight(xs, ss, f=0.5):

    M = xs.shape[0]
    N = xs.shape[1]
    w = np.zeros(N)

    for m in range(M):
        w += ss[m]*(xs[m]-f)*(1-f)

    return w

def weight_matrix(xs, ss, f=0.5):

    M = xs.shape[0]

    w = ss[0] * np.outer(xs[0] - f, xs[0] - f)

    if len(ss) > 1:
        if np.all(ss[1:-1] == ss[2:]) and ss[1] == 1:
            w += np.einsum("ni,nj->ij", xs - f, xs - f)
            w -= np.outer(xs[0] - f, xs[0] - f)

        else:
            for m in range(M - 1):
                w += ss[m + 1] * np.outer(xs[m + 1] - f, xs[m + 1] - f)

    return w


def pseudo_matrix(xs, ss, f=0.5):

    N = xs.shape[1]
    S = np.sum(ss * ss) - ss[0] * ss[0]
    s_sum = np.sum(ss)

    w_pre = np.reshape(np.random.normal(0, np.sqrt(S) * f * (1 - f), N * N), (N, N))
    w = np.triu(w_pre) + np.triu(w_pre, 1).T

    w += ss[0] * np.outer(xs[0] - f, xs[0] - f)

    for n in range(N):
        w[n, n] = s_sum * 0.25

    return w

