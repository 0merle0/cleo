#!/usr/bin/env python3
"""Recover the MoMI library's fragment boundaries and per-part distances.

Nothing in the data states where the five slots start and stop, so they are
MEASURED rather than assumed. The parent is the positive control --
`1.g20___2.g20___3.g20___4.g20___5.g20`, 52 replicate wells of one sequence,
and no sample keeps all five parent slots. Take the designs that swap exactly
ONE slot away from it and see which positions move; those positions are that
slot's span.

    slot 1    1-40      from 714 single-swap designs
    slot 2   45-84      from 719
    slot 3   89-126     from 716
    slot 4  131-163     from 721
    slot 5  168-209     from 606

The gaps -- 41-44, 85-88, 127-130, 164-167 -- are positions no design ever
mutates, so the true cut points lie inside them and the windows above are the
tightest bounds the data supports. The slots come out non-overlapping and in
order, which is a check on the method rather than an assumption of it.

THE CATALYTIC TRIAD SPANS TWO SLOTS: Ser66 on slot 2, His107 and Asp122 on
slot 3. Recombining this library reassembles the catalytic machinery across a
cut, which is why retention is worth measuring at all.

Writes data/momi_fragment_parts.csv: one row per distinct part, with how far
it sits from the G20 part in the same slot.

    uv run python paper/figures/momi_fragments.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "momi_libraries.csv"
OUT = HERE / "data" / "momi_fragment_parts.csv"
K = 5


def load():
    d = pd.read_csv(SRC)
    ctrl = d[d.sample_type == "positive_control"]
    if ctrl.sequence.nunique() != 1:
        sys.exit("positive controls are not a single sequence")
    s = d[d.sample_type == "sample"].copy()
    s["parts"] = s.name.astype(str).str.split(":").str[0].str.split("___")
    return ctrl.sequence.iloc[0], s


def spans(parent, s):
    """-> [(lo, hi)] per slot, from single-slot swaps against the parent."""
    out = []
    for i in range(K):
        one = s[s.parts.apply(
            lambda ps: ("g20" not in ps[i]) and
                       all("g20" in p for j, p in enumerate(ps) if j != i))]
        pos = set()
        for seq in one.sequence:
            pos |= {j for j, (a, b) in enumerate(zip(seq, parent)) if a != b}
        if not pos:
            sys.exit(f"no single-swap designs for slot {i + 1}")
        out.append((min(pos), max(pos)))
    for i in range(K - 1):
        if out[i][1] >= out[i + 1][0]:
            sys.exit(f"slots {i+1} and {i+2} overlap -- check the parent")
    return out


def main():
    parent, s = load()
    sp = spans(parent, s)
    print(f"parent {len(parent)} aa; slot windows "
          + ", ".join(f"{lo+1}-{hi+1}" for lo, hi in sp))

    rows, seen = [], {}
    for lib, parts, seq in zip(s.library_name, s.parts, s.sequence):
        for i, pid in enumerate(parts):
            key = (i, pid)
            if key in seen:
                seen[key][2].add(lib)
                continue
            lo, hi = sp[i]
            n = sum(1 for a, b in zip(seq[lo:hi + 1], parent[lo:hi + 1])
                    if a != b)
            seen[key] = [pid, n, {lib}]
    for (i, pid), (_, n, libs) in seen.items():
        rows.append(dict(slot=i + 1, part=pid, n_mut_from_g20=n,
                         is_parent=int("g20" in pid),
                         libraries="+".join(sorted(libs))))
    f = pd.DataFrame(rows).sort_values(["slot", "n_mut_from_g20"])
    f.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(f)} distinct parts")
    nz = f[f.is_parent == 0]
    print("\nmutations from G20 per part, by slot (parent part excluded):")
    print(nz.groupby("slot").n_mut_from_g20
          .agg(["count", "mean", "min", "max"]).round(1).to_string())
    print("\nparts shared between the two libraries:")
    print(f.libraries.value_counts().to_string())


if __name__ == "__main__":
    main()
