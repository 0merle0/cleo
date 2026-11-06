"""Snapshot the PETase prediction-vs-activity join into paper/figures/data/.

Sources live outside this repo (two other projects, 6.6 MB of wide CSV); the
paper needs ~3k rows and a dozen columns. Provenance is recorded in the header
of the written file and in paper/figures/data/README_petase.md.
"""
import csv

P = "/home/jgershon/projects"
UNI = f"{P}/adaptyv_library/rosetta/activity_unified.csv"
BF = f"{P}/itopt/kiera/af3_petase/big_fold/metrics_wide.csv"
OUT = "/home/jgershon/git/cleo/paper/figures/data/petase_activity_zeroshot.csv"

# The parent of the design lineage (209 aa, Ser66/His107/Asp122). Taken from
# momi_final_redesign/curate_library.py, where it is the reference sequence the
# RL leash is measured against. It was never itself assayed in this table, so it
# is a distance origin only -- not an activity comparator.
PARENT = ("MGWEEELWLERPSGERTPVRRHHFPAKKAENAEEAIANVERLIEEIEASGITFTATADRQVVVGWSLGVI"
          "TGMIMHATGTDFITALRKALEIGKKVKKEDPEFMQRHKKIVTDGNRAEIREDIDYWIEVIYKETGYPVDR"
          "SRIFIAETVEEAVELAREAVELGHAIIHLPPTLIGTAGEYVAEWLASEGVDVLLMGGLGSGVPVTLYQA")

# The 20 designs re-assayed by Adaptyv after selection from the screen. Same
# assay, different run (see adaptyv_library/rosetta/build_unified_and_tasks.py),
# so they are flagged rather than pooled.
REVALIDATED = {f"s0{n}" for n in range(3371, 3391)}

METRICS = [
    "acyl_his107_ne2_plddt", "acyl_activesite_plddt", "acyl_ser66_og_plddt",
    "acyl_pae_ser66_his107", "acyl_ptm", "acyl_prot_lig_iptm",
    "substrate_ptm", "substrate_iptm", "substrate_interface_pae_min",
]

bf = {r["seqid"]: r for r in csv.DictReader(open(BF))}
rows = []
for r in csv.DictReader(open(UNI)):
    a, s = r["momi_unified"], r["sequence"]
    if not a or len(s) != len(PARENT):
        continue
    b = bf.get(r["seqid"], {})
    out = {
        "seqid": r["seqid"],
        "activity": a,
        "n_mut_from_parent": sum(1 for x, y in zip(s, PARENT) if x != y),
        "momi_round": b.get("momi_round", ""),
        "revalidated": int(r["seqid"] in REVALIDATED),
    }
    out.update({m: b.get(m, "") for m in METRICS})
    rows.append(out)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT}: {len(rows)} rows")
print("with AF3 metrics:", sum(1 for r in rows if r["acyl_activesite_plddt"]))
print("revalidated:", sum(r["revalidated"] for r in rows))
