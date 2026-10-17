# paper/ — start here

There are four documents in this directory that could each be mistaken for "the
outline". They are not interchangeable. This file says which one answers which
question, so the next person does not have to open all four to find out.

## If you want...

| ...the | read | what it is |
|---|---|---|
| **outline of the paper** | **`plan_combinatorial.md`** | The thesis in one paragraph, then R1–R5: the five results in the order the paper makes them, each tagged measured / running / partial. **This is the outline.** |
| **numbers to put in a slide** | **`key_results.md`** | Every headline figure with the script that regenerates it and a status of solid / provisional / missing. Written to be read top to bottom before a talk. |
| **structure of the LaTeX build** | `docindex.tex` | Which `.tex` file holds what, and each one's drafting status. Renders into the PDF in draft mode (p. 2), so it is also the reader's map. |
| **experimental protocol** | `experiments_plan.md` | The E1–E21 slate: configs, arms, pre-registrations, what each experiment would have to show to count. Long, and a record rather than a plan now. |

Everything else is narrower: `transfer_gap.md` is one 2×2 explained,
`spec_modularity_rl.md` is one proposed experiment, `figure_plan.md` is stale
(August) but its per-panel "what would make this unconvincing" notes are still
worth reading.

## The three documents that build

```
make pdf        main.tex              the paper
make notebook   notebook.tex          the AME experimental record (E1-E21)
make library    library_notebook.tex  the library arc (L1-L5)
make all        all three
```

**The paper and the record are deliberately separate documents.** The notebook
keeps pre-registrations next to the results that confirmed or killed them,
including the ones that got retracted; the paper carries only what survived.
Nothing in `notebook.tex` or `library_notebook.tex` is `\input` by `main.tex`.

## The spine of the paper, in three moves

1. **`results/frontier.tex`** — function-aligned RL moves the diversity/fidelity
   frontier. Public benchmark (RFdiffusion2 AME), referee-reproducible.
2. **`results/fragments.tex`** — fragmentation and recombination turn that
   frontier into a combinatorially large *physical* library.
3. **`results/experimental.tex`** — that library, ordered and assayed. 3,135
   recombined PETase designs; the oracle calibrated against measured function.

Moves 1 and 2 are the method; move 3 is the experimental check on both.

## Figures

Scripts in `figures/`, exporting **SVG only** — `make` derives
`figures/pdf/` with Inkscape at build time, so the PDFs are disposable and the
SVG is the source of truth. Small curated inputs live in `figures/data/` with a
README recording provenance; large source tables stay in their own projects and
are snapshotted by a script rather than read at build time.

## Archive

`archive/` holds the wet-lab application arcs (heme, protease), a superseded
outline, and the *plan* for a three-round closed-loop PETase campaign that has
no data. `archive/README.md` records why each moved and what had to be rewritten
to remove it without breaking the build. Nothing is deleted; `git log --follow`
reaches every file at its original path.

> The archived PETase **plan** and the live `results/experimental.tex` are easy
> to confuse. The plan is about optimizing an enzyme across rounds and its
> headline "250x" was always a placeholder. The live section is about whether
> the library is physically realizable and whether the oracle predicts function.
