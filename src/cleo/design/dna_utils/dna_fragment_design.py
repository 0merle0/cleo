#!/software/containers/universal.sif
'''
# Fragment Ordering
 - Written by Stacey Gerben srgerb@uw.edu using Ryan Kibler's Domesticator
 - take in fasta or fastas of amino acid sequences
 - ensure that the start is labeled differently, i.e. 1-4
 - group by set
 - check for consistent overlap
 - separate overlaps into other group
 - reverse translate all groups (this can be done with multiprocessing)
 - find compatable overhangs in overlaps
 - combine all
     - vec_adapter_5' + 1.x + first_half_of_consistent_overlap_1_w_overhang
     - second_half_of_consistent_overlap_1_w_overhang + 2.x + first_half_of_consistent_overlap_2_w_overhang
     - ...
     - second_half_of_consistent_overlap_last_w_overhang + last.x + vec_adapter_5'
Example ways to call:
frag_ordering_clean.py --csv final_fragments_to_order_stacey.csv
frag_ordering_clean.py --fasta final_fragments_to_order_stacey.fa

'''
# In[39]:


from Bio import SeqIO, SeqUtils, Seq, SeqFeature
import re
from datetime import date
import getpass

import pandas as pd
import json
import math
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import copy
import numpy as np
from collections import Counter

# Domesticator uses dnachisel
import dnachisel
from dnachisel import DnaOptimizationProblem, NoSolutionError
from dnachisel import DEFAULT_SPECIFICATIONS_DICT
from dnachisel import Location
from dnachisel import Specification, SpecEvaluation

today = date.today().strftime("%Y%m%d")[2:]


# In[ ]:





# In[40]:


def get_arguments(argv=None):
    parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter,
            description=" * Generates IPDblocks with adapters based on orthogonal primers and outputs echo and ordering files\n"
            )
    # REQUIRED
    parser.add_argument(
            '--fasta', '-f',
            help='fasta file or files to parse',
            action='store',
            type=str,
            nargs='+'
            )
    parser.add_argument(
            '--csv', '-i',
            help='csv to parse',
            action='store',
            type=str,
            nargs='+'
            )
    parser.add_argument(
            '--vector_json_path',
            help='Total volume in PCR1 in uL',
            action='store',
            type=str,
            default="/software/lab/johnbercow/entry_vectors/vectors_enzymes.json",
            )
    parser.add_argument(
            '--vector', '-p',
            help='name of vector plasmid',
            action='store',
            type=str,
            default="CF_1",
            )
    parser.add_argument(
            '--overlap_len', '-l',
            help='how many amino acids are conserved between your fragments',
            action='store',
            type=int,
            default="4",
            )
    parser.add_argument(
            '--fidelity',
            help='csv comparing fidelity data. Default is for BsaI TODO: find for other commonly used enzymes and add all to a folder',
            action='store',
            type=str,
            default='/net/shared/IPDblocks/files/b4-BsaI-HFv2-37_16_cycling.table-overhang_matrix.csv',
            )
    parser.add_argument(
            '-o',
            help='info_in_output_file',
            action='store',
            type=str,
            default='frag_dna_output',
            )
    args = parser.parse_args(argv)
    return args


# In[41]:


def get_vector_enzyme(file_path = "/net/software/lab/johnbercow/entry_vectors/vectors_enzymes_seq.json"):
    with open(file_path, 'r') as file:
        vector_enzyme = json.load(file)
    vectors = vector_enzyme['vectors']
    cuts = vector_enzyme['enzymes']
    return vectors, cuts
def reverse_complement(seq):
    complement = {'A':'T','T':'A','C':'G','G':'C'}
    return ''.join(complement[x] for x in reversed(seq.upper()))


# In[42]:


# Special k-mer minimasation objective from Ryan's domesticator.
class MinimizeNumKmers(Specification):
    """Minimizes a kmer score."""

    best_possible_score = 0

    def __init__(self, k=8, location=None, boost=1.0):
        self.location = location
        self.k = k
        self.boost = boost

    def initialize_on_problem(self, problem, role=None):
        return self._copy_with_full_span_if_no_location(problem)

    def evaluate(self, problem):
        """Return a customized kmer score for the problem's sequence"""
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
        """String representation."""
        return "MinimizeNum%dmers" % self.k

DEFAULT_SPECIFICATIONS_DICT["MinimizeNumKmers"] = MinimizeNumKmers

# ============================================
# FUNCTIONS
# ============================================
def reverse_translate(
        amino_acid_sequence,
        kmers_weight=1.0,
        cai_weight=1.0,
        hairpins_weight=1.0,
        max_tries=10,
        species='e_coli',
        avoid=['GGTCTC', 'GAGACC']
    ):
    '''
    Ryan's domesticator.
    '''

    # Generate a naively reverse-translated DNA sequence first.
    naive_dna_sequence = dnachisel.reverse_translate(amino_acid_sequence)

    # Sequence optimisation will happen across the whole sequence.
    location = Location.from_biopython_location(SeqFeature.FeatureLocation(0, len(amino_acid_sequence) * 3))

    # Add optimisation objectives.
    objectives = []
    objectives.append(MinimizeNumKmers(k=8, boost=kmers_weight, location=location))
    objectives.append(dnachisel.builtin_specifications.AvoidHairpins(boost=hairpins_weight, location=location))
    objectives.append(dnachisel.builtin_specifications.MaximizeCAI(species=species, boost=cai_weight, location=location))

    # Add optimisation constraints.
    constraints = []
    constraints.append(dnachisel.builtin_specifications.EnforceTranslation(location=location))

    # A series of sequence patterns to remove
    constraints.append(dnachisel.builtin_specifications.AvoidPattern("AAAAA", location=location)) # terminator
    constraints.append(dnachisel.builtin_specifications.AvoidPattern("TTTTT", location=location)) # terminator
    constraints.append(dnachisel.builtin_specifications.AvoidPattern("CCCCCC", location=location)) # repetitions
    constraints.append(dnachisel.builtin_specifications.AvoidPattern("GGGGGG", location=location)) # repetitions
    constraints.append(dnachisel.builtin_specifications.AvoidPattern("ATCTGTT", location=location)) # T7/T3 RNA polymerase pausing
    constraints.append(dnachisel.builtin_specifications.AvoidPattern("GGRGGT", location=location)) # G-quadruplex?

    for seq in avoid: # GG enzyme recognition site.
        constraints.append(dnachisel.builtin_specifications.AvoidPattern(seq, location=location))

    # Organism-specific constraints
    if species == 'e_coli':
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("GGAGG", location=location)) # E. coli Shine-Dalgarno
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("TAAGGAG", location=location)) # strong RBS
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("GCTGGTGG", location=location)) # Chi site in E. coli

        # Alternative start sites: G/A rich 6 nt upstream of ATG or GTG or TTG (cryptic start sites)
        # N = A or T or C or G
        # D = A or G or T
        # R = A or G
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC523762/?page=1
        # This constraint can be too harsh and lead to problems with some sequences.
        constraints_easier = copy.deepcopy(constraints)
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("RRRRRNNNNNDTG", location=location)) # 5 nt spacing
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("RRRRRNNNNNNDTG", location=location)) # 6 nt spacing
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("RRRRRNNNNNNNDTG", location=location)) # 7 nt spacing

    elif species == 'h_sapiens':
        constraints.append(dnachisel.builtin_specifications.AvoidPattern("GCCRCCATGG", location=location)) # Kozak sequence

    # Global GC content (according to TWIST).
    constraints.append(dnachisel.builtin_specifications.EnforceGCContent(mini=0.25, maxi=0.65, location=location))

    # Local GC content (according to TWIST).
    constraints.append(dnachisel.builtin_specifications.EnforceGCContent(mini=0.35, maxi=0.65, window=50, location=location))

    # Start optimisation.
    solutions = []
    solution_found = False
    for i in range(max_tries):

        if solution_found:
            break

        try:
            if species == 'e_coli' and i >= max_tries/2:
                #print('  [!] Preventing alternative start sites removed from the list of optimisation constraints.')
                initial_problem = DnaOptimizationProblem(naive_dna_sequence, constraints=constraints_easier, objectives=objectives, logger=None)

            else:
                initial_problem = DnaOptimizationProblem(naive_dna_sequence, constraints=constraints, objectives=objectives, logger=None)

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

    # Return the best solution according to dnachisel if multiple attemptes were made.
    scores = [solution.objectives_evaluations().scores_sum() for solution in solutions]
    best_idx = np.argmin(scores)

    best_solution = solutions[best_idx]

    return best_solution.sequence


# In[49]:


def read_input_files(args):
    seq_dict = {'fragment_name':[],'fragment_sequence':[],'fragment_number':[]}
    if args.fasta:
        delimiters_for_split = '[\._|-]'
        for fasta in args.fasta:
            for record in SeqIO.parse(fasta, "fasta"):
                frag_set = re.split(delimiters_for_split,record.id)[0]
                seq_dict['fragment_name'].append(record.id)
                seq_dict['fragment_sequence'].append(record.seq)
                seq_dict['fragment_number'].append(frag_set)
    df = pd.DataFrame.from_dict(seq_dict)
    if args.csv:
        for csv in args.csv:
            df2 = pd.read_csv(csv)
            df = pd.concat([df, df2], ignore_index=True)
    df['fragment_number'] = df['fragment_number'].astype(int)
    df = df.sort_values(by='fragment_number', ignore_index=True)
    df.rename(columns={"fragment_sequence": "fragment_aa_sequence"})
    return df

def find_overlaps_and_add_cut_aa_seqs_to_df(df, overlap_len):
    overlaps = {}
    overlap_len = math.floor(args.overlap_len/2)
    ordered_set = df['fragment_number'].unique()
    for i, frag_set in enumerate(df['fragment_number'].unique()):
        temp_df = df[df['fragment_number'] == frag_set].copy().reset_index(drop=True)
        if i < len(ordered_set) - 1:
            first_seq = temp_df['fragment_sequence'][0] # get first sequence in this set
            start_ol = first_seq[-1*overlap_len:]
            overlaps[f'{i}_{i+1}'] = start_ol # get last amino acids of that sequence
            for j,r in temp_df.iterrows():
                assert r.fragment_sequence[-1*overlap_len:] == start_ol 
        if i != 0:
            first_seq = temp_df['fragment_sequence'][0] # get first sequence in this set
            end_ol = first_seq[:overlap_len]
            overlaps[f'{i-1}_{i}'] += end_ol # get last amino acids of that sequence
            for j,r in temp_df.iterrows():
                assert r.fragment_sequence[:overlap_len] == end_ol
    cut_sequences = []
    for i, r in df.iterrows():
        if r.fragment_number == ordered_set[0]:
            cut_sequences.append(r.fragment_sequence[:-1*overlap_len])
        elif r.fragment_number == ordered_set[len(ordered_set) - 1]:
            cut_sequences.append(r.fragment_sequence[overlap_len:])
        else:
            cut_sequences.append(r.fragment_sequence[overlap_len:-1*overlap_len])
    df['cut_aa_sequences'] = cut_sequences
    return df, overlaps

def find_cutsites(overlaps, vector_info, sticky, cycles=5, max_off_target=0):
    dna_overlaps={}
    overhangs = [vector_info["5'-sticky"][:sticky].upper(),vector_info["3'-sticky"][-1*sticky:].upper(), reverse_complement(vector_info["5'-sticky"][:sticky].upper()),reverse_complement(vector_info["3'-sticky"][-1*sticky:].upper())]
    found_all_ol = False
    while found_all_ol == False and cycles > 0:
        found_all_ol = True
        cycles -= 1
        for key in overlaps.keys():
            if found_all_ol == False:
                break
            ol = reverse_translate(overlaps[key])
            for i in range(len(ol)-sticky):
                good_sticky = True
                temp_overhang = ol[i:i+sticky].upper()
                if fidelity_df[temp_overhang][temp_overhang] > max_off_target:
                    good_sticky = False
                for item in overhangs:
                    if fidelity_df[temp_overhang][item] > max_off_target:
                        good_sticky = False
                        break
                if good_sticky:
                    dna_overlaps[key] = (ol[:i+sticky],ol[i:])
                    overhangs.append(temp_overhang)
                    overhangs.append(reverse_complement(temp_overhang))
                    break
                if i == len(ol)-sticky-1:
                    found_all_ol = False
                    break
    return dna_overlaps

def combine_dna(df,dna_overlaps,vector_info,cuts):
    full_dna = []
    frag_len = []
    fw_cut, rv_cut, spacer, sticky = cuts
    rand_seq = 'atactacggtctcacgagaccgtaatgc'
    fw_adapter = f'{fw_cut}{rand_seq[:spacer]}'
    rv_adapter = f'{rand_seq[-1*spacer:]}{rv_cut}'
    ordered_set = df['fragment_number'].unique()
    for i, frag in enumerate(ordered_set):
        temp_df = df[df['fragment_number'] == frag].copy().reset_index(drop=True)
        for j, r in temp_df.iterrows():
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

def to_fasta(df,filename):
    with open(f'{filename}.fa','w') as fasta:
        for i,r in df.iterrows():
            fasta.write(f'>{r.fragment_name}\n{r.full_dna}\n')


# In[44]:


#args = get_arguments(argv=["--csv=/home/srgerb/scripts/frag_ordering_ipdblocks/final_fragments_to_order_stacey.csv" ])
args = get_arguments()


# In[31]:


if __name__ == "__main__":
    # get info from inputs
    fidelity_df = pd.read_csv(args.fidelity, index_col = 'Overhang')
    vectors, cuts = get_vector_enzyme(file_path = "/software/lab/johnbercow/entry_vectors/vectors_enzymes_seq.json")
    vector_info = vectors[args.vector]
    enzyme = vector_info['Enzyme']
    fw_cut, rv_cut, spacer, sticky = cuts[enzyme]
    df = read_input_files(args)
    
    # find overlaps and reverse translate
    df, overlaps = find_overlaps_and_add_cut_aa_seqs_to_df(df,args.overlap_len/2)
    dna_overlaps = find_cutsites(overlaps, vector_info, sticky)
    cut_seqs = df.cut_aa_sequences.to_list()
    chunksize = max(1, len(cut_seqs) // (cpu_count() * 4))
    with Pool() as pool:
        dna_seqs = list(
            tqdm(
                pool.imap(reverse_translate, cut_seqs, chunksize=chunksize),
                total=len(cut_seqs)
            )
        )
    df['cut_dna_seq'] = dna_seqs
    df_final = combine_dna(df,dna_overlaps,vector_info,cuts[enzyme])
    filename = f'{today}_{getpass.getuser()}_{args.o}'
    df_final.to_csv(f'{filename}.csv')
    to_fasta(df_final,filename)
    print(f"Longest fragment length : {df_final['frag_len'].max()}")

