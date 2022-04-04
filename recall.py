import numpy as np


def gibbs(W, M, x_t, s_t, update_x, update_s, samples=100):
    x = np.copy(x_t)
    s = np.copy(s_t)
    N = len(x_t)

    x_samples = np.empty((samples + 1, N))
    x_samples[0] = x
    s_samples = np.empty((samples + 1))
    s_samples[0] = s

    for c in range(samples):
        idx = np.arange(N)
        np.random.shuffle(idx)

        s = update_s(W, x, M, s)

        for k in idx:
            x = update_x(W, x, x_t, s, k, M)

        x_samples[c + 1] = x
        s_samples[c + 1] = s

    return x_samples, s_samples

def gibbs_flip(W, M, x_0, x_t, s_t, update_x_flip, update_s, samples=100):
    x = np.copy(x_0)
    s = np.copy(s_t)
    N = len(x_t)

    x_samples = []
    x_samples.append(x)
    s_samples = []
    s_samples.append(s)

    record = True

    for c in range(samples):
        idx = np.arange(N)
        np.random.shuffle(idx)

        s = update_s(W, x, M, s)

        for i in range(int(N/2)):

            k1 = idx[i]
            k2 = idx[i+1]

            if x[k1] != x[k2]:
                record = True
                x = update_x_flip(W, x, x_t, s, k1, k2, M)

        if record:
            x_samples.append(x)
            s_samples.append(s)
            #record = False

    return x_samples, s_samples





def gibbs_ff(W, W_ff, M, x_t, s_t, update_x, update_s, samples=100):
    x = np.copy(x_t)
    s = np.copy(s_t)
    N = len(x_t)

    x_samples = np.empty((samples + 1, N))
    x_samples[0] = x
    s_samples = np.empty((samples + 1))
    s_samples[0] = s

    for c in range(samples):
        idx = np.arange(N)
        np.random.shuffle(idx)

        s = update_s(W_ff, x, M, s)

        for k in idx:
            x = update_x(W, x, x_t, s, k, M)

        x_samples[c + 1] = x
        s_samples[c + 1] = s

    return x_samples, s_samples


def maps(W, M, x_t, s_t, update_x, update_s, samples=100):
    x = np.copy(x_t)
    s = s_t
    N = len(x_t)

    for c in range(samples):
        x_old = np.copy(x)
        idx = np.arange(N)
        np.random.shuffle(idx)

        s = update_s(W, x, M, s)

        for k in idx:
            x = update_x(W, x, x_t, s, k, M)


        if np.sum((x + x_old) % 2) == 0:
            return np.array([x, ] * samples), np.array([s, ] * samples), c

    return np.array([x, ] * samples), np.array([s, ] * samples), samples

def maps_info(W, M, x_t, s_t, update_x, update_s, x_T, samples=100):
    x = np.copy(x_t)
    s = s_t
    N = len(x_t)
    errors = np.zeros(samples+1)
    changes = np.zeros(samples)

    errors[0] = np.sum((x + x_T) % 2)

    for c in range(samples):
        x_old = np.copy(x)
        idx = np.arange(N)
        np.random.shuffle(idx)

        s = update_s(W, x, M, s)

        for k in idx:
            x = update_x(W, x, x_t, s, k, M)

        errors[c+1] = np.sum((x + x_T) % 2)
        changes[c] = np.sum((x + x_old) % 2)

        if np.sum((x + x_old) % 2) == 0:
            return np.array([x, ] * samples), np.array([s, ] * samples), errors, changes

    return np.array([x, ] * samples), np.array([s, ] * samples), errors, changes

def maps_steps(W, M, x_t, s_t, update_x, update_s, samples=100, steps=100):
    x = np.copy(x_t)
    s = s_t
    N = len(x_t)
    x_log = np.zeros([samples, N])
    s_log = np.zeros(samples)
    i_log = 0
    x_log[i_log] = x
    s_log[i_log] = s

    for c in range(samples):
        x_old = np.copy(x)
        idx = np.arange(N)
        np.random.shuffle(idx)

        for k in idx:
            if (k % steps == 0) and (k != 0):
                s = update_s(W, x, M, s)
                if c == 0:
                    s = (s+s_t)/2
                i_log += 1
                x_log[i_log] = x
                s_log[i_log] = s

            x = update_x(W, x, x_t, s, k, M)

        s = update_s(W, x, M, s)

        i_log += 1
        if i_log > samples:
            i_log = samples - int(samples/steps) - 1
        x_log[i_log] = x
        s_log[i_log] = s


        if np.sum((x + x_old) % 2) == 0:
            x_log[i_log:] = np.array([x, ] * (samples - i_log))
            s_log[i_log:] = np.array([s, ] * (samples - i_log))

            return x_log, s_log, c

    x_log[i_log:] = np.array([x, ] * (samples - i_log))
    s_log[i_log:] = np.array([s, ] * (samples - i_log))

    return x_log, s_log, c


def maps_s(W, M, x_t, s_t, update_x, update_s, samples=100):
    x = np.copy(x_t)
    s = s_t
    N = len(x_t)

    for c in range(samples):
        x_old = np.copy(x)
        idx = np.arange(N)
        np.random.shuffle(idx)

        for k in idx:
            x = update_x(W, x, x_t, s, k, M)

        s = update_s(W, x, M, s)

        if np.sum((x + x_old) % 2) == 0:
            return np.array([x, ] * samples), np.array([s, ] * samples), c

    return np.array([x, ] * samples), np.array([s, ] * samples), samples


def maps2(W, M, x_t, s_t, update_x, update_s, samples=100):
    x = np.copy(x_t)
    s = s_t
    N = len(x_t)


    x_samples = []
    s_samples = []
    x_samples.append(x)

    for c in range(samples):
        x_old = np.copy(x)
        idx = np.arange(N)
        np.random.shuffle(idx)

        s = update_s(W, x, M, s)

        for k in idx:
            x = update_x(W, x, x_t, s, k, M)

        x_samples.append(x)
        if np.sum((x + x_old) % 2) == 0:
            for i in range(c, samples):
                x_samples.append(x)
                s_samples.append(s)

            s_samples.append(s)
            return np.array(x_samples), np.array(s_samples), samples

    s_samples.append(s)
    return np.array(x_samples), np.array(s_samples), samples

