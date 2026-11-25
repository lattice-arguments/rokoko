#!/usr/bin/env python
import argparse
import itertools
import math
import sys
from multiprocessing import Pool

HEADER = "cost_model,shape,zeta,n,m,q,bound,status,log2_rop,beta,d,zeta_out,log2_repetitions"

Q20, Q23, Q31, Q32, Q34, Q50 = 2**20 - 3, 8380417, 2**31 - 159, 2**32 - 1119, 2**34 - 351, 2**50 - 27
POW2 = [2**k for k in range(9, 18)]


def product(models, zetas, ranks, widths, moduli, bounds):
    return [
        (cost_model, shape, zeta, n, m, q, bound)
        for (cost_model, shape), zeta, q, bound, n, m in itertools.product(models, zetas, moduli, bounds, ranks, widths)
    ]


def grid_full():
    target = [("adps16", "lgsa")]
    models = [("adps16", "lgsa"), ("adps16", "gsa"), ("matzov", "lgsa"), ("matzov", "gsa")]
    rows = product(target, [0], range(14, 129), POW2 + [768, 3000, 50000], [Q20, Q23, Q31, Q32, Q34, Q50], [1])
    rows += product(target, [0], [14, 16, 24, 32, 48, 64, 96, 128], [512, 2048, 8192, 32768], [Q20, Q31, Q50], [2, 8, 32])
    rows += product(target, [0], [2, 4, 8], [64, 512, 4096], [Q20, Q31], [1, 32, 1000])
    rows += product(models, [0, 100], [14, 24, 48, 96], [512, 2048, 8192], [Q20, Q31, Q50], [1, 2, 8, 32])
    rows += product(models, ["opt"], [16, 32, 64], [512, 2048], [Q20, Q31], [1, 2, 8, 32])
    rows += product(models, [0, "opt"], [113], [494, 1000], [2048], [1, 16, 80])
    rows += [
        (cost_model, shape, zeta, n, m, q, bound)
        for (cost_model, shape), zeta in itertools.product(models, [0, "opt"])
        for n, m, q, bound in [
            (60, 500, 1024, 32),
            (200, 1500, 2048, 80),
            (50, 300, 512, 10),
            (150, 1500, 8192, 100),
            (256, 2048, Q23, 32),
            (512, 4096, 2**30, 10),
            (1024, 2304, Q23, 350209),
            (1536, 3072, Q23, 724481),
        ]
    ]
    return rows


def grid_pinned():
    target = [("adps16", "lgsa")]
    models = [("adps16", "lgsa"), ("adps16", "gsa"), ("matzov", "lgsa"), ("matzov", "gsa")]
    rows = [("adps16", "lgsa", 0, n, m, Q31, 1) for n, m in [(16, 2048), (24, 2048), (32, 2048), (32, 4096), (48, 4096)]]
    rows += product(target, [0], [14, 32, 79, 80, 128], [512, 4096, 32768, 131072], [Q31], [1])
    rows += product(target, [0], [16, 48, 112], [1024, 3000, 16384], [Q20, Q32, Q34, Q50], [1])
    rows += product(target, [0], [24, 64], [512, 8192], [Q20, Q50], [2, 8, 32])
    rows += product(target, [0], [2], [64, 512], [Q20], [1, 1000])
    rows += product(models, [0, 100], [24, 96], [512, 8192], [Q31], [1, 8])
    rows += product(models, ["opt"], [16, 64], [512], [Q20, Q31], [1, 32])
    rows += product(models, [0, "opt"], [113], [494], [2048], [16])
    rows += product(models, [0, "opt"], [200], [1500], [2048], [80])
    rows += product(models, [0, "opt"], [1024], [2304], [Q23], [350209])
    return rows


AMPLIFY = [-0.02, -0.5, -1, -3.3, -10, -26.4, -27, -52, -53.5, -100, -1000, -1024.5]
AMPLIFY += [-2040.3, -2044.7, -2046.5, -2047.5, -2048.99, -2049, -2049.01]


def amplify(log2_success):
    from estimator import prob
    from sage.all import RR

    return f"{log2_success},{log2(prob.amplify(0.99, 2 ** RR(log2_success)))}"


def log2(x):
    from sage.all import RealField, oo

    return math.inf if x == oo else float(RealField(200)(x).log2())


def estimate(row, explicit_n):
    import estimator.sis_lattice as sis_lattice
    from estimator import RC, SIS
    from estimator.reduction import ADPS16, ReductionCost
    from sage.all import RR, floor, oo

    # upstream's symbolic erf can fail in the mpmath/gmpy2 conversion; same formula in double precision
    sis_lattice.gaussian_cdf = lambda mu, sigma, t: RR(0.5 * (1 + math.erf(float(t - mu) / (math.sqrt(2) * float(sigma)))))

    def short_vectors(self, beta, d, **kwds):
        return ReductionCost._short_vectors_sieve(self, beta, d, N=floor(2 ** RR(0.2075 * beta)), **kwds)

    ADPS16.short_vectors = short_vectors if explicit_n else ReductionCost._short_vectors_sieve
    sis_lattice.SISLattice.cost_infinity.clear_cache()

    cost_model, shape, zeta, n, m, q, bound = row
    params = SIS.Parameters(n=n, q=q, length_bound=bound, m=m, norm=oo)
    cost = SIS.lattice(
        params,
        zeta=None if zeta == "opt" else zeta,
        red_cost_model={"adps16": RC.ADPS16, "matzov": RC.MATZOV}[cost_model],
        red_shape_model=shape,
    )
    fields = [log2(cost["rop"])] + [cost.get(k, "-") for k in ("beta", "d", "zeta")]
    return fields + [log2(cost["repetitions"]) if "repetitions" in cost else "-"]


def reference(args):
    row, explicit_n = args
    status = "ok"
    try:
        fields = estimate(row, explicit_n)
    except OverflowError:
        status = "overflow"
        try:
            fields = estimate(row, True)
        except Exception as error:
            status, fields = type(error).__name__, ["-"] * 5
    except Exception as error:
        status, fields = type(error).__name__, ["-"] * 5
    return ",".join(str(x) for x in (*row, status, *fields))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", default="lattice-estimator")
    parser.add_argument("--grid", choices=["pinned", "full", "amplify"], required=True)
    parser.add_argument("--explicit-n", action="store_true")
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.estimator)

    if args.grid == "amplify":
        print("log2_success,log2_trials", *map(amplify, AMPLIFY), sep="\n")
        sys.exit()

    rows = {"pinned": grid_pinned, "full": grid_full}[args.grid]()
    print(HEADER)
    with Pool(args.jobs) as pool:
        for line in pool.imap(reference, [(row, args.explicit_n) for row in rows], chunksize=4):
            print(line, flush=True)
