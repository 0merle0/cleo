"""What post-training costs, in folds, against brute-force filtering.

THE FOLD BILL IS FLAT AND ONE-TIME. A run is 200 steps; build_chimeras emits
rotations * batch chimeras per step and each is folded best-of-5, so

    folds/step = rotations x batch x samples = 4 x 16 x 5 = 320
    200 steps                                             = 64,000 folds

which the run configs corroborate: the 150-step k=4 anchor is logged at
48,000 folds, and batch 32 at 128 chimeras/step is logged at 107,520 for 336
steps. Cost is flat in k -- cutting finer does not fold more -- so one number
covers the whole sweep.

THE COMPARISON. The alternative to training is to fold your way to a good
library: generate candidates with stock ProteinMPNN, fold each one, keep the
ones that pass. To end up with S members that fold, at vanilla pass rate p,
you must fold S/p candidates at 5 folds each:

    brute force   =  5 S / p        folds, LINEAR in library size
    post-training =  64,000         folds, INDEPENDENT of library size

The second is flat because the enrichment lives in the policy, not in the
members: once trained, every design drawn from it inherits the higher pass
rate without being folded.

WHAT THIS DOES AND DOES NOT SAY. It is an argument about scaling, not a claim
that training is always cheaper. Below the crossover, brute force wins and the
honest thing is to say so -- at M0097's 59.8% vanilla rate the crossover is
about 7.7k members. The point is that a combinatorial library is not a few
thousand members: k=4 with 100 parts per slot is 10^8, where brute-force
filtering would cost ~10^9 folds against the same 64,000.
"""

import numpy as np

FOLDS_PER_STEP = 4 * 16 * 5        # rotations x batch x best-of-5
TRAIN_STEPS = 200
TRAIN_FOLDS = FOLDS_PER_STEP * TRAIN_STEPS
SAMPLES = 5                        # best-of-5, the AME benchmark's own unit


def brute_force_folds(S, p):
    """-> folds needed to filter S passing members out of stock MPNN."""
    return SAMPLES * np.asarray(S, dtype=float) / p


def crossover(p):
    """-> library size at which training costs the same as brute force."""
    return TRAIN_FOLDS * p / SAMPLES


def report():
    """Print the numbers the text quotes, so they are not retyped by hand."""
    rates = {"M0097": 59.8, "M0255": 25.8, "M0315": 49.6}
    print(f"training, one time: {TRAIN_FOLDS:,} folds "
          f"({FOLDS_PER_STEP}/step x {TRAIN_STEPS} steps)")
    print("  cross-check: 150-step k=4 anchor logged at 48,000; "
          "batch 32 at 128 chimeras/step logged at 107,520 for 336 steps")
    print("\nbreak-even against brute-force filtering:")
    for site, pct in rates.items():
        print(f"  {site}  vanilla {pct:4.1f}%  ->  "
              f"{crossover(pct / 100):,.0f} members")
    print("\nbrute force at scale (M0097's 59.8%):")
    for S in (1e4, 1e6, 1e8, 1e12):
        f = brute_force_folds(S, 0.598)
        print(f"  S={S:>8.0e}  {f:>20,.0f} folds   "
              f"{f / TRAIN_FOLDS:>14,.0f}x training")


if __name__ == "__main__":
    report()
