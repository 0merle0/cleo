"""Resolve ``${...}`` input templates for a dataset example (SPEC 6.3).

Inputs declared by a reward package are **templates** resolved per-example against
named sources. There is **no implicit fallback** — every source is named explicitly,
and any unsatisfied argument is a **hard error** naming the row ``id``, the package,
and the offending argument. A dataset that loads is guaranteed that every example
fully satisfies its reward's declared contract (``requires`` + ``steps.*.inputs``).

Sources (SPEC 6.3 table):
    ``${row.<field>...}``   a (possibly nested) field of the JSONL row, e.g.
                            ``${row.params.epitope_residues}``
    ``${native.seq}``       native seq of the DESIGN chain (``design_chain``)
    ``${native.seq.<ch>}``  native seq of a named chain, e.g. ``${native.seq.T}``
    ``${design.seq}``       the sampled sequence under score (sentinel ``DESIGN_SEQ``;
                            already present as the df ``sequence`` column)
    literal                 any non-template value (str without ``${}``, or non-str)

Resolved values are seeded as **df columns** by the reward registry's
``get_input_df`` (SPEC 6.4) — the exact mechanism ``protenix_from_df`` already reads
(``antigen_sequence`` / ``antigen_msa_dir`` / ``epitope_residues``).
"""
from __future__ import annotations

import re

from cleo.design.data.mask import as_chain_list

_TEMPLATE = re.compile(r"^\$\{([^}]+)\}$")

#: sentinel: the value is the per-sample designed sequence (df ``sequence`` column)
DESIGN_SEQ = object()


class BindingError(ValueError):
    """Raised when a template cannot be resolved or a required input is unsatisfied."""


def is_template(value) -> bool:
    return isinstance(value, str) and _TEMPLATE.match(value.strip()) is not None


def resolve_one(template, row: dict, native_seq_fn):
    """Resolve a single template (or pass a literal through unchanged).

    Args:
        template: a ``${...}`` string, or a literal value returned as-is.
        row: the JSONL example row (dict).
        native_seq_fn: ``callable(chain_or_None) -> str`` — resolves ``native.seq[.chain]``;
            ``None`` means the design chain. Raises ``BindingError`` on a missing chain.
    """
    rid = row.get("id", "<no-id>")
    m = _TEMPLATE.match(template.strip()) if isinstance(template, str) else None
    if m is None:
        return template  # literal (non-template string, number, list, ...)

    path = m.group(1).strip()
    parts = path.split(".")
    head = parts[0]

    if head == "design":
        if parts[1:] == ["seq"]:
            return DESIGN_SEQ
        raise BindingError(f"[{rid}] unsupported design template ${{{path}}} (only ${{design.seq}})")

    if head == "native":
        if len(parts) >= 2 and parts[1] == "seq":
            chain = parts[2] if len(parts) >= 3 else None
            return native_seq_fn(chain)
        raise BindingError(f"[{rid}] unsupported native template ${{{path}}} (only ${{native.seq[.<chain>]}})")

    if head == "row":
        cur = row
        for key in parts[1:]:
            if not isinstance(cur, dict) or key not in cur:
                raise BindingError(f"[{rid}] template ${{{path}}} references missing field {key!r}")
            cur = cur[key]
        return cur

    raise BindingError(f"[{rid}] unknown template source {head!r} in ${{{path}}}")


def collect_inputs(package) -> dict[str, object]:
    """Flatten every step's ``inputs`` map into one ``{column: template}`` dict.

    All steps share one df, so per-example inputs are seeded as columns up front
    regardless of which step consumes them. A later step re-declaring the same
    column with a different template is a package authoring error -> raise.
    """
    merged: dict[str, object] = {}
    for step in package.get("steps", []) or []:
        inputs = step.get("inputs", {}) or {}
        for col, tmpl in inputs.items():
            if col in merged and merged[col] != tmpl:
                raise BindingError(
                    f"reward package declares column {col!r} with conflicting templates "
                    f"{merged[col]!r} vs {tmpl!r}"
                )
            merged[col] = tmpl
    return merged


def resolve_inputs(package, row: dict, native_seq_fn) -> dict[str, object]:
    """Resolve all declared inputs for a row and enforce the ``requires`` contract.

    Returns ``{column: resolved_value}`` (value may be the ``DESIGN_SEQ`` sentinel).
    Raises ``BindingError`` if any template is malformed/unresolvable, or if a name
    in ``requires`` is not among the resolved inputs (or resolves to ``None``).
    """
    rid = row.get("id", "<no-id>")
    templates = collect_inputs(package)
    resolved = {col: resolve_one(tmpl, row, native_seq_fn) for col, tmpl in templates.items()}

    for name in package.get("requires", []) or []:
        if name not in resolved:
            raise BindingError(
                f"[{rid}] reward package requires {name!r} but no step declares it under inputs"
            )
        if resolved[name] is None:
            raise BindingError(f"[{rid}] required input {name!r} resolved to None")
    return resolved


def native_seq_fn(row: dict, resolver):
    """Build the ``native_seq_fn`` closure: maps ``None`` -> design chain, then delegates.

    ``resolver`` is ``callable(chain) -> str`` (raises ``BindingError`` if absent) —
    a light presence check at validation time, or the real sequence at bind time.
    """
    design_chains = as_chain_list(row["design_chain"])

    def fn(chain):
        c = chain if chain is not None else design_chains[0]
        return resolver(c)

    return fn
