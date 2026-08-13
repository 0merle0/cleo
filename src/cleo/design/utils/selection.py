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
}
