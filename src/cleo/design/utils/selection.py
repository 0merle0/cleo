"""Selection rules for choosing which sequences to fold or to order.

Sampling from MPNN is nearly free; folding is essentially the entire cost. So
the question "which N of these M sequences do we spend folds on?" is worth
optimising directly. This module implements the candidate rules and a common
distance backend so they can be compared on equal terms.

Distances are Hamming by default. Pass ``embeddings`` (one row per sequence,
e.g. mean-pooled ESM) to score in embedding space instead -- Hamming answers
"how many positions differ", an embedding answers "how biochemically
different", and the two can rank candidates differently.

MEASURED, on the M0097 trajectory pool (3200 designs, 97 passing, labels known):
pure ``maxmin`` is *worse than random* at every fold budget tested (at 800
folds: 51 passing as-sampled, 25 random, 6 max-min). Farthest-point traversal
walks to the extremes of the pool, and extreme sequences fail. Any rule used to
decide what to fold has to defend itself against random on passing yield, not
on diversity -- diversity is easy and worthless on its own. ``anchor_band`` and
``cluster_rep`` exist because both bound how far the selection is allowed to
stray, which is the specific failure being corrected.
"""

import numpy as np

try:                                       # optional; only cluster_rep needs it
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
except ImportError:                        # pragma: no cover
    linkage = None


def encode(seqs):
    """-> (n, L) byte array of equal-length sequences."""
    seqs = list(seqs)
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("sequences must be equal length")
    return np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(len(seqs), L)


def consensus(seqs):
    """Per-position modal residue -- the low-temperature mode as one sequence."""
    M = encode(seqs)
    return "".join(np.unique(M[:, c], return_counts=True)[0]
                   [np.unique(M[:, c], return_counts=True)[1].argmax()].decode()
                   for c in range(M.shape[1]))


class Distances:
    """Row-to-set distances, Hamming or embedding-space, with one interface."""

    def __init__(self, seqs, embeddings=None):
        self.M = encode(seqs)
        self.E = None if embeddings is None else np.asarray(embeddings, float)
        if self.E is not None and len(self.E) != len(self.M):
            raise ValueError("embeddings must have one row per sequence")

    def __len__(self):
        return len(self.M)

    def to_index(self, i):
        """Distance from every sequence to sequence i."""
        if self.E is None:
            return (self.M != self.M[i]).sum(1).astype(np.float64)
        return np.linalg.norm(self.E - self.E[i], axis=1)

    def to_seq(self, seq, embedding=None):
        """Distance from every sequence to an external reference."""
        if self.E is None or embedding is None:
            return (self.M != encode([seq])[0]).sum(1).astype(np.float64)
        return np.linalg.norm(self.E - np.asarray(embedding, float), axis=1)


def random_select(D, k, seed=0):
    """Uniform without replacement. The control every other rule must beat."""
    return np.random.default_rng(seed).choice(len(D), min(k, len(D)), replace=False)


def maxmin(D, k, seed=0, anchor_d=None, w_self=1.0, w_anchor=0.0, seeded=None):
    """Greedy farthest-point (k-center) traversal.

    Scores each candidate by its MINIMUM distance to everything already picked
    -- the min, not the mean, is what stops a tight cluster far from the rest
    from being accepted wholesale. ``anchor_d`` mixes in distance from a fixed
    reference (the low-T consensus), and ``seeded`` continues an existing
    library rather than starting a new one.
    """
    n, k = len(D), min(k, len(D))
    if seeded is not None and len(seeded):
        dmin = np.min([D.to_index(i) for i in seeded], axis=0)
        sel = []
    else:
        first = int(np.random.default_rng(seed).integers(n))
        sel, dmin = [first], D.to_index(first)
    while len(sel) < k:
        score = w_self * dmin
        if anchor_d is not None:
            score = score + w_anchor * anchor_d
        score[sel] = -np.inf
        i = int(score.argmax())
        sel.append(i)
        dmin = np.minimum(dmin, D.to_index(i))
    return np.array(sel)


def anchor_band(D, k, anchor_d, lo=0.4, hi=0.8, seed=0):
    """Select inside a distance shell around the anchor, diversifying within it.

    The fix for max-min's failure mode. Rather than maximising distance without
    limit -- which lands on outliers that do not fold -- restrict candidates to
    those whose distance from the low-temperature consensus falls between the
    ``lo`` and ``hi`` quantiles of the pool, then run max-min inside that shell.
    Far enough from the incumbent mode to be a genuinely new solution, close
    enough to remain a plausible protein.
    """
    a, b = np.quantile(anchor_d, [lo, hi])
    pool = np.flatnonzero((anchor_d >= a) & (anchor_d <= b))
    if len(pool) <= k:
        return pool
    sub = _Subset(D, pool)
    return pool[maxmin(sub, k, seed=seed)]


def cluster_rep(D, k, seed=0, thresh=None):
    """Cluster the pool and take one representative each, cutting to give ~k.

    Bounded by construction: representatives are drawn from across the pool's
    density rather than from its edges, so unlike max-min this cannot spend the
    whole budget on outliers.
    """
    if linkage is None:
        raise ImportError("cluster_rep needs scipy")
    n = len(D)
    if k >= n:
        return np.arange(n)
    full = np.array([D.to_index(i) for i in range(n)])
    Z = linkage(squareform((full + full.T) / 2, checks=False), "average")
    lab = fcluster(Z, t=k, criterion="maxclust") if thresh is None else \
        fcluster(Z, t=thresh, criterion="distance")
    rng = np.random.default_rng(seed)
    out = [rng.choice(np.flatnonzero(lab == c)) for c in np.unique(lab)]
    return np.array(out[:k])


def logprob_strat(logp, k, seed=0, n_bins=None):
    """Stratify the pool by the policy's own log-probability, sample evenly.

    Every other rule here scores on sequence distance alone. That is what sank
    ``maxmin`` in E16: it walked to sequences that were distant *because they
    were degenerate*, and folded eight times fewer passing designs than random.
    The policy's log-probability of each sampled sequence is already computed
    during the rollout and costs nothing to reuse, and it is a quality signal
    that no distance-based rule has access to.

    The objective here is deliberately not library diversity. GRPO's advantage
    is group-relative: it divides by the spread of reward *within the group*, so
    a batch whose rewards are nearly identical carries almost no gradient
    regardless of how good the designs are. Stratifying by log-probability
    targets that directly -- it guarantees the group spans the policy's range
    rather than concentrating wherever the policy currently puts its mass --
    while keeping the high-probability mode represented, which is the part
    ``maxmin`` threw away.

    `logp` is the per-sequence total log-probability over designed positions.
    Bins are equal-count quantiles of the pool, so the rule adapts to whatever
    the log-prob distribution looks like at this step rather than assuming it is
    stationary across training (it is not -- it sharpens as the policy converges).
    """
    logp = np.asarray(logp, float)
    n = len(logp)
    if k >= n:
        return np.arange(n)
    bins = int(n_bins or min(k, 8))
    rng = np.random.default_rng(seed)

    # Equal-count strata by rank, so bin membership does not depend on the
    # scale or shape of the log-prob distribution.
    order = np.argsort(-logp, kind="mergesort")          # best first
    strata = np.array_split(order, bins)

    # Deal k picks round-robin across strata so the remainder is spread rather
    # than dumped on one bin, then draw without replacement inside each.
    take = np.full(bins, k // bins)
    take[: k % bins] += 1
    out = []
    for s, t in zip(strata, take):
        t = min(t, len(s))
        out.extend(rng.choice(s, t, replace=False) if t else [])
    out = np.array(out, dtype=int)

    # Round-robin can under-fill when a stratum is smaller than its quota.
    if len(out) < k:
        rest = np.setdiff1d(np.arange(n), out, assume_unique=False)
        out = np.concatenate([out, rng.choice(rest, k - len(out), replace=False)])
    return out[:k]


def logprob_top(logp, k, **_):
    """Greedy top-k by policy log-probability. Pure exploitation.

    The opposite extreme from ``maxmin``, and included for exactly that reason:
    it brackets the axis. If quality-directed selection helps at all this is
    where it shows, and if it collapses the group's reward spread to nothing
    then GRPO should stall here even though every folded design is good --
    which is the cleanest available test of whether the group-relative
    baseline, rather than design quality, is the binding constraint.
    """
    return np.argsort(-np.asarray(logp, float), kind="mergesort")[:k]


def logprob_band(D, logp, k, lo=0.25, hi=1.0, seed=0):
    """Drop the low-log-prob tail, then run max-min inside what is left.

    The targeted repair of E16's failure. ``maxmin`` was not wrong about
    diversity -- it delivered 1.43x more distinct substitutions per passing
    design -- it was wrong about which sequences were available to be diverse
    *in*. Bounding the pool below by policy log-probability removes the
    degenerate sequences that farthest-point traversal is otherwise drawn to,
    while leaving the traversal itself untouched.

    `lo`/`hi` are quantiles of the pool's own log-prob distribution, so the
    band tracks the policy as it sharpens rather than pinning to a fixed value.
    """
    logp = np.asarray(logp, float)
    n = len(logp)
    if k >= n:
        return np.arange(n)
    qlo, qhi = np.quantile(logp, [lo, hi])
    keep = np.flatnonzero((logp >= qlo) & (logp <= qhi))
    if len(keep) < k:                       # band too tight at this step
        keep = np.argsort(-logp, kind="mergesort")[:max(k, 1)]
    sub = maxmin(_Subset(D, keep), k, seed=seed)
    return keep[sub]


class _Subset:
    """Restrict a Distances object to a subset of rows, keeping the interface."""

    def __init__(self, D, idx):
        self.D, self.idx = D, np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def to_index(self, i):
        return self.D.to_index(self.idx[i])[self.idx]


RULES = {
    "random": lambda D, k, **kw: random_select(D, k, kw.get("seed", 0)),
    "maxmin": lambda D, k, **kw: maxmin(D, k, seed=kw.get("seed", 0)),
    "maxmin_anchor": lambda D, k, **kw: maxmin(
        D, k, seed=kw.get("seed", 0), anchor_d=kw["anchor_d"],
        w_self=kw.get("w_self", 1.0), w_anchor=kw.get("w_anchor", 1.0)),
    "anchor_band": lambda D, k, **kw: anchor_band(
        D, k, kw["anchor_d"], kw.get("lo", 0.4), kw.get("hi", 0.8),
        kw.get("seed", 0)),
    "anchor_far": lambda D, k, **kw: np.argsort(-kw["anchor_d"])[:k],
    "cluster_rep": lambda D, k, **kw: cluster_rep(D, k, kw.get("seed", 0)),
    # Log-prob-directed rules. These need `logp` (per-sequence total log
    # probability over designed positions), which the caller takes from the
    # rollout output -- it is already computed there and costs nothing to reuse.
    "logprob_strat": lambda D, k, **kw: logprob_strat(
        kw["logp"], k, seed=kw.get("seed", 0), n_bins=kw.get("n_bins")),
    "logprob_top": lambda D, k, **kw: logprob_top(kw["logp"], k),
    "logprob_band": lambda D, k, **kw: logprob_band(
        D, kw["logp"], k, lo=kw.get("lo", 0.25), hi=kw.get("hi", 1.0),
        seed=kw.get("seed", 0)),
}
