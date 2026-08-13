"""Mean-pooled ESM-2 embeddings for sequence sets.

Hamming distance answers "how many positions differ". It cannot tell a
conservative substitution from a disruptive one, so two libraries with equal
Hamming spread can be very unequal in how much biochemical ground they cover.
This module supplies the second view.

Embeddings are mean-pooled over residues (excluding the BOS/EOS tokens and any
padding) to give one vector per sequence, which is what the downstream uses
need -- PCA panels, and optionally distances in
``cleo.design.utils.selection.Distances``.

Deliberately NOT the primary metric. Every quantitative diversity claim in the
paper is on Hamming, so that the figures and tables measure the same thing; an
embedding distance is a diagnostic and a secondary library metric, reported
alongside rather than instead. Swapping the primary metric to ESM would also
make the numbers depend on a checkpoint, which is a dependency a benchmark
comparison does not need.

Requires ``transformers``; install with ``uv add transformers`` or
``uv sync --extra esm``. Import is lazy so the rest of the package does not
need it.

    uv run python -m cleo.design.utils.embedding \
        --seqs experiments/ame/bestofn/grpo_windows.csv \
        --out embeddings.npz --batch-size 8
"""

import argparse
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


def embed_sequences(seqs, model_name=DEFAULT_MODEL, batch_size=8, device=None,
                    fp16=True, progress=True):
    """-> (n, d) float32 array of mean-pooled per-sequence embeddings.

    Pooling masks out padding *and* the BOS/EOS tokens ESM adds; including them
    drags every vector toward a shared constant and compresses the distances we
    are trying to measure, by an amount that depends on sequence length.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    seqs = list(seqs)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        # ESM checkpoints carry no pooler; AutoModel would otherwise attach a
        # randomly initialised one and warn about it. We never read
        # pooler_output -- pooling happens below over last_hidden_state -- so
        # the head is pure noise in the logs, but suppressing the warning by
        # not creating the weights is better than teaching readers to ignore it.
        model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    except TypeError:
        model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()
    if fp16 and device.startswith("cuda"):
        model = model.half()

    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            h = model(**enc).last_hidden_state          # (b, T, d)

            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                              # BOS
            lengths = enc["attention_mask"].sum(1) - 1
            mask[torch.arange(len(batch), device=device), lengths] = 0  # EOS
            m = mask.unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
            if progress:
                print(f"  embedded {min(i + batch_size, len(seqs))}/{len(seqs)}",
                      flush=True)
    return np.concatenate(out, 0)


def load_cached(path):
    """-> (sequences, embeddings). Companion to the CLI's .npz output."""
    z = np.load(path, allow_pickle=True)
    return list(z["sequences"]), z["embeddings"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", required=True, help="CSV with a `sequence` column")
    ap.add_argument("--out", required=True, help=".npz of sequences + embeddings")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    # Embed each distinct sequence once; pools contain heavy duplication across
    # training steps and the model is the entire cost here.
    seqs = pd.read_csv(a.seqs).sequence.drop_duplicates().tolist()
    print(f"{len(seqs)} unique sequences -> {a.model}")
    E = embed_sequences(seqs, a.model, a.batch_size, a.device, not a.no_fp16)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, sequences=np.array(seqs, object), embeddings=E)
    print(f"wrote {E.shape} -> {a.out}")


if __name__ == "__main__":
    main()
