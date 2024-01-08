import numpy as np
import rusvm


class SmoothMax2:
    def __init__(self, smoothing):
        self.smoothing = smoothing

    def __call__(self, t):
        s = self.smoothing
        return np.piecewise(t, [
            t >= s,
            t <= -s,
        ], [
            lambda x: x,
            lambda x: 0.0,
            lambda x: 0.25 / s * (x + s) ** 2,
        ])

    def deriv(self, t):
        s = self.smoothing
        return np.piecewise(t, [
            t >= s,
            t <= -s,
        ], [
            lambda x: 1.0,
            lambda x: 0.0,
            lambda x: 0.5 * (x + s) / s,
        ])

    def deriv2(self, t):
        s = self.smoothing
        return np.piecewise(t, [
            t >= s,
            t <= -s,
        ], [
            lambda x: 0.0,
            lambda x: 0.0,
            lambda x: 0.5 / s,
        ])


def main():
    # generate sample dataset
    n, nft = 2000, 2
    np.random.seed(0)
    x = np.random.rand(n, nft)
    t = x.sum(axis=1)
    # y = np.where(t > t.mean(), 1.0, -1.0)
    y = np.round(5 * t ** 2)

    lmbda = 1e-3
    smoothing = 0.1
    epsilon = 0.5
    kind = 'lssvm'

    res = rusvm.solve_smonewt(
        x, y,
        params_problem=dict(
            kind=kind,
            lmbda=lmbda,
            # smoothing=smoothing,
            # max_asum=5.0,
            # epsilon=epsilon,
        ),
        params_smo=dict(
            verbose=1000,
            log_objective=True,
            max_steps=500000,
            time_limit=30.0,
            # max_steps=1_000_000,
            # shrinking_period=1000,
            tol=1e-3,
            # second_order=False,
            cache_size=n,
        ),
        params_newton=dict(
            verbose=1,
            max_steps=500,
            # max_steps=1_000_000,
            # shrinking_period=10000,
            tol=1e-12,
        ),
        # callback=callback,
    )
    b = res['b']
    c = res['c']
    a = np.array(res['a'])
    ka = np.array(res['ka'])

    maxfun = SmoothMax2(smoothing)

    if kind == 'classification':
        dec = ka + b + c * y
        t = 1.0 - y * dec
        dmax = y * maxfun.deriv(t)
        print(np.hstack((dmax[:, None], a[:, None])))
    elif kind == 'regression':
        s = np.concatenate((np.ones_like(y), -np.ones_like(y)))
        dec = ka + b + c * s
        yy = np.concatenate((y, y))
        t = yy - dec
        dmax = s * maxfun.deriv(s * t - epsilon)
        print(np.hstack((dmax[:, None], a[:, None])))


if __name__ == '__main__':
    main()
