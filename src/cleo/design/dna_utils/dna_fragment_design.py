"""
Fragment DNA Design
 - Written by Stacey Gerben srgerb@uw.edu using Ryan Kibler's Domesticator
 - take in fasta or csv of amino acid fragments
 - group by fragment number
 - check for consistent overlap between adjacent fragments
 - reverse translate all groups (multiprocessing)
 - find compatible overhangs in overlaps
 - combine with vector adapters and output DNA sequences

Usage:
  python -m cleo.design.dna_utils.dna_fragment_design --config-name dna_fragment_design
"""

import os
import re
import json
import math
import copy
import getpass
from pathlib import Path
from datetime import date
from multiprocessing import Pool, cpu_count
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO, SeqFeature
from tqdm import tqdm
import hydra

import dnachisel
from dnachisel import DnaOptimizationProblem, NoSolutionError
from dnachisel import DEFAULT_SPECIFICATIONS_DICT
from dnachisel import Location
from dnachisel import Specification, SpecEvaluation


# ============================================
# Custom dnachisel specification
# ============================================

class MinimizeNumKmers(Specification):
    """Minimizes a kmer score. From Ryan Kibler's Domesticator."""

    best_possible_score = 0

    def __init__(self, k=8, location=None, boost=1.0):
        self.location = location
        self.k = k
        self.boost = boost

    def initialize_on_problem(self, problem, role=None):
        return self._copy_with_full_span_if_no_location(problem)

    def evaluate(self, problem):
        sequence = self.location.extract_sequence(problem.sequence)
        all_kmers = [sequence[i : i + self.k] for i in range(len(sequence) - self.k)]
        number_of_non_unique_kmers = sum(
            [count for kmer, count in Counter(all_kmers).items() if count > 1]
        )
        score = -(float(self.k) * number_of_non_unique_kmers) / len(sequence)
        return SpecEvaluation(
            self,
            problem,
            score=score,
            locations=[self.location],
            message="Score: %.02f (%d non-unique %d-mers)"
            % (score, number_of_non_unique_kmers, self.k),
        )

    def label_parameters(self):
        return [("k", str(self.k))]

    def short_label(self):
        return f"Avoid {self.k}mers {self.boost}"

    def __str__(self):
        return "MinimizeNum%dmers" % self.k

DEFAULT_SPECIFICATIONS_DICT["MinimizeNumKmers"] = MinimizeNumKmers


# ============================================
# Core functions
# ============================================

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[x] for x in reversed(seq.upper()))


def reverse_translate(
        amino_acid_sequence,
        kmers_weight=1.0,
        cai_weight=1.0,
        hairpins_weight=1.0,
        max_tries=10,
        species='e_coli',
        avoid=None,
    ):
    """
    Codon-optimize a protein sequence to DNA using Ryan Kibler's Domesticator.
    Enforces translation, avoids problematic motifs, and optimizes for CAI,
    k-mer diversity, and hairpin avoidance.
    """
    if avoid is None:
        avoid = ['GGTCTC', 'GAGACC']

    naive_dna_sequence = dnachisel.reverse_translate(str(amino_acid_sequence))
    location = Location.from_biopython_location(
        SeqFeature.FeatureLocation(0, len(amino_acid_sequence) * 3)
    )

    objectives = [
        MinimizeNumKmers(k=8, boost=kmers_weight, location=location),
        dnachisel.builtin_specifications.AvoidHairpins(boost=hairpins_weight, location=location),
        dnachisel.builtin_specifications.MaximizeCAI(species=species, boost=cai_weight, location=location),
    ]

    constraints = [
        dnachisel.builtin_specifications.EnforceTranslation(location=location),
        dnachisel.builtin_specifications.AvoidPattern("AAAAA", location=location),
        dnachisel.builtin_specifications.AvoidPattern("TTTTT", location=location),
        dnachisel.builtin_specifications.AvoidPattern("CCCCCC", location=location),
        dnachisel.builtin_specifications.AvoidPattern("GGGGGG", location=location),
        dnachisel.builtin_specifications.AvoidPattern("ATCTGTT", location=location),
        dnachisel.builtin_specifications.AvoidPattern("GGRGGT", location=location),
    ]

    for seq in avoid:
        constraints.append(dnachisel.builtin_specifications.AvoidPattern(seq, location=location))

    if species == 'e_coli':
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("GGAGG", location=location))
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("TAAGGAG", location=location))
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("GCTGGTGG", location=location))
        constraints_easier = copy.deepcopy(constraints)
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("RRRRRNNNNNDTG", location=location))
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("RRRRRNNNNNNDTG", location=location))
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("RRRRRNNNNNNNDTG", location=location))
    elif species == 'h_sapiens':
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("GCCRCCATGG", location=location))
        constraints_easier = copy.deepcopy(constraints)

    constraints.append(dnachisel.builtin_specifications.EnforceGCContent(mini=0.25, maxi=0.65, location=location))
    constraints.append(dnachisel.builtin_specifications.EnforceGCContent(mini=0.35, maxi=0.65, window=50, location=location))

    solutions = []
    solution_found = False
    for i in range(max_tries):
        if solution_found:
            break
        try:
            if species == 'e_coli' and i >= max_tries / 2:
                initial_problem = DnaOptimizationProblem(
                    naive_dna_sequence, constraints=constraints_easier,
                    objectives=objectives, logger=None
                )
            else:
                initial_problem = DnaOptimizationProblem(
                    naive_dna_sequence, constraints=constraints,
                    objectives=objectives, logger=None
                )
            problem = copy.deepcopy(initial_problem)
            problem.resolve_constraints_by_random_mutations()
            problem.optimize()
            problem.resolve_constraints(final_check=True)
            solutions.append(problem)
            solution_found = True
        except NoSolutionError:
            initial_problem.max_random_iters += 1000
            solution_found = False
            continue

    if len(solutions) == 0:
        raise NoSolutionError(f"No solution found for {amino_acid_sequence}", initial_problem)

    scores = [solution.objectives_evaluations().scores_sum() for solution in solutions]
    best_idx = np.argmin(scores)
    return solutions[best_idx].sequence


def get_vector_enzyme(file_path):
    with open(file_path, 'r') as f:
        vector_enzyme = json.load(f)
    return vector_enzyme['vectors'], vector_enzyme['enzymes']


def read_input_files(fasta_paths=None, csv_paths=None):
    """Read fragment sequences from FASTA and/or CSV files.

    FASTA headers must follow the convention:  {frag_num}.{unique_id}
    where frag_num is a 1-indexed integer (e.g. "1.0003.a7b2c8d1").
    The first dot/underscore/hyphen-delimited token is parsed as the
    integer fragment number used for grouping and ordering.
    """
    seq_dict = {'fragment_name': [], 'fragment_sequence': [], 'fragment_number': []}

    if fasta_paths:
        delimiters_for_split = r'[\._|-]'
        for fasta_path in fasta_paths:
            for record in SeqIO.parse(fasta_path, "fasta"):
                frag_set = re.split(delimiters_for_split, record.id)[0]
                seq_dict['fragment_name'].append(record.id)
                seq_dict['fragment_sequence'].append(str(record.seq))
                seq_dict['fragment_number'].append(frag_set)

    df = pd.DataFrame.from_dict(seq_dict)

    if csv_paths:
        for csv_path in csv_paths:
            df2 = pd.read_csv(csv_path)
            df = pd.concat([df, df2], ignore_index=True)

    df['fragment_number'] = df['fragment_number'].astype(int)
    df = df.sort_values(by='fragment_number', ignore_index=True)
    return df


def find_overlaps_and_add_cut_aa_seqs_to_df(df, overlap_len):
    """
    Check that adjacent fragments have consistent overlaps of the given length
    (in amino acids). Trim overlap regions from fragment sequences and return
    the overlap sequences for later DNA cutsite design.
    """
    half_overlap = math.floor(overlap_len / 2)
    overlaps = {}
    ordered_set = df['fragment_number'].unique()

    for i, frag_set in enumerate(ordered_set):
        temp_df = df[df['fragment_number'] == frag_set].copy().reset_index(drop=True)

        if i < len(ordered_set) - 1:
            first_seq = temp_df['fragment_sequence'][0]
            start_ol = first_seq[-1 * half_overlap:]
            overlaps[f'{i}_{i+1}'] = start_ol
            for _, r in temp_df.iterrows():
                assert r.fragment_sequence[-1 * half_overlap:] == start_ol, \
                    f"Inconsistent overlap at end of fragment {frag_set}"

        if i != 0:
            first_seq = temp_df['fragment_sequence'][0]
            end_ol = first_seq[:half_overlap]
            overlaps[f'{i-1}_{i}'] += end_ol
            for _, r in temp_df.iterrows():
                assert r.fragment_sequence[:half_overlap] == end_ol, \
                    f"Inconsistent overlap at start of fragment {frag_set}"

    cut_sequences = []
    for _, r in df.iterrows():
        if r.fragment_number == ordered_set[0]:
            cut_sequences.append(r.fragment_sequence[:-1 * half_overlap])
        elif r.fragment_number == ordered_set[-1]:
            cut_sequences.append(r.fragment_sequence[half_overlap:])
        else:
            cut_sequences.append(r.fragment_sequence[half_overlap:-1 * half_overlap])

    df['cut_aa_sequences'] = cut_sequences
    return df, overlaps


def find_cutsites(overlaps, vector_info, sticky, fidelity_df, cycles=5, max_off_target=0):
    """Find compatible Golden Gate overhangs within the overlap regions."""
    dna_overlaps = {}
    overhangs = [
        vector_info["5'-sticky"][:sticky].upper(),
        vector_info["3'-sticky"][-1 * sticky:].upper(),
        reverse_complement(vector_info["5'-sticky"][:sticky].upper()),
        reverse_complement(vector_info["3'-sticky"][-1 * sticky:].upper()),
    ]

    found_all_ol = False
    while not found_all_ol and cycles > 0:
        found_all_ol = True
        cycles -= 1
        for key in overlaps.keys():
            if not found_all_ol:
                break
            ol = reverse_translate(overlaps[key])
            for i in range(len(ol) - sticky):
                good_sticky = True
                temp_overhang = ol[i:i + sticky].upper()
                if fidelity_df[temp_overhang][temp_overhang] > max_off_target:
                    good_sticky = False
                for item in overhangs:
                    if fidelity_df[temp_overhang][item] > max_off_target:
                        good_sticky = False
                        break
                if good_sticky:
                    dna_overlaps[key] = (ol[:i + sticky], ol[i:])
                    overhangs.append(temp_overhang)
                    overhangs.append(reverse_complement(temp_overhang))
                    break
                if i == len(ol) - sticky - 1:
                    found_all_ol = False
                    break

    return dna_overlaps


def combine_dna(df, dna_overlaps, vector_info, cuts):
    """Add vector adapters and overlap DNA to each fragment's reverse-translated DNA."""
    full_dna = []
    frag_len = []
    fw_cut, rv_cut, spacer, sticky = cuts
    rand_seq = 'atactacggtctcacgagaccgtaatgc'
    fw_adapter = f'{fw_cut}{rand_seq[:spacer]}'
    rv_adapter = f'{rand_seq[-1 * spacer:]}{rv_cut}'
    ordered_set = df['fragment_number'].unique()

    for i, frag in enumerate(ordered_set):
        temp_df = df[df['fragment_number'] == frag].copy().reset_index(drop=True)
        for _, r in temp_df.iterrows():
            if i == 0:
                fragment = fw_adapter + vector_info["5'-sticky"] + r.cut_dna_seq + dna_overlaps[f'{i}_{i+1}'][0] + rv_adapter
            elif i == len(ordered_set) - 1:
                fragment = fw_adapter + dna_overlaps[f'{i-1}_{i}'][1] + r.cut_dna_seq + vector_info["3'-sticky"] + rv_adapter
            else:
                fragment = fw_adapter + dna_overlaps[f'{i-1}_{i}'][1] + r.cut_dna_seq + dna_overlaps[f'{i}_{i+1}'][0] + rv_adapter
            full_dna.append(fragment)
            frag_len.append(len(fragment))

    df['full_dna'] = full_dna
    df['frag_len'] = frag_len
    return df


def to_fasta(df, filepath):
    with open(filepath, 'w') as f:
        for _, r in df.iterrows():
            f.write(f'>{r.fragment_name}\n{r.full_dna}\n')


# ============================================
# Hydra entrypoint
# ============================================

_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../../config/design")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="dna_fragment_design")
def main(cfg):

    today_str = date.today().strftime("%Y%m%d")[2:]
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load vector/enzyme info
    vectors, cuts = get_vector_enzyme(cfg.vector_json_path)
    vector_info = vectors[cfg.vector]
    enzyme = vector_info['Enzyme']
    fw_cut, rv_cut, spacer, sticky = cuts[enzyme]

    # Load fidelity data
    fidelity_df = pd.read_csv(cfg.fidelity_csv, index_col='Overhang')

    # Read input fragments
    fasta_paths = list(cfg.fasta) if cfg.get("fasta") else None
    csv_paths = list(cfg.csv) if cfg.get("csv") else None
    assert fasta_paths or csv_paths, "Must provide at least one of 'fasta' or 'csv' input paths"

    print("Reading input files...")
    df = read_input_files(fasta_paths=fasta_paths, csv_paths=csv_paths)
    print(f"  Loaded {len(df)} fragments across {df['fragment_number'].nunique()} regions")

    # Find overlaps and trim
    print(f"Finding overlaps (overlap_len={cfg.overlap_len})...")
    df, overlaps = find_overlaps_and_add_cut_aa_seqs_to_df(df, cfg.overlap_len)
    print(f"  Found {len(overlaps)} overlap regions")

    # Find compatible Golden Gate cutsites
    print("Finding compatible cutsites...")
    dna_overlaps = find_cutsites(overlaps, vector_info, sticky, fidelity_df)
    print(f"  Found {len(dna_overlaps)} cutsite pairs")

    # Reverse translate fragment sequences (multiprocessing)
    cut_seqs = df.cut_aa_sequences.tolist()
    chunksize = max(1, len(cut_seqs) // (cpu_count() * 4))
    print(f"Reverse translating {len(cut_seqs)} fragments (species={cfg.species})...")
    with Pool() as pool:
        dna_seqs = list(
            tqdm(
                pool.imap(reverse_translate, cut_seqs, chunksize=chunksize),
                total=len(cut_seqs)
            )
        )
    df['cut_dna_seq'] = dna_seqs

    # Combine with adapters
    print("Combining with vector adapters...")
    df_final = combine_dna(df, dna_overlaps, vector_info, cuts[enzyme])

    # Save outputs
    filename_base = os.path.join(cfg.output_dir, f"{today_str}_{getpass.getuser()}_{cfg.output_name}")
    df_final.to_csv(f"{filename_base}.csv", index=False)
    to_fasta(df_final, f"{filename_base}.fa")

    print(f"\nWrote {len(df_final)} DNA fragments")
    print(f"  CSV: {filename_base}.csv")
    print(f"  FASTA: {filename_base}.fa")
    print(f"  Longest fragment: {df_final['frag_len'].max()} bp")
    print("Done.")


if __name__ == "__main__":
    main()
