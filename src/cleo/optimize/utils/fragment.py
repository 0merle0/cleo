"""Fragment dictionary utilities for combinatorial library optimization.

Provides functions for parsing fragment CSV files into dictionaries,
enumerating all possible combinatorial sequences, and featurizing
fragment-based sequence names into integer index tensors.
"""
import torch
import pandas as pd


def get_fragment_dictionary(fragment_csv):
    """Parse a fragment CSV into an ordered dictionary of (name, sequence) pairs.

    The CSV must have columns ``name``, ``fragment`` (1-indexed integer region
    number), and ``seq``. Entries within each fragment are sorted alphabetically
    by name for deterministic ordering.

    Args:
        fragment_csv: Path to the CSV file.

    Returns:
        Dict mapping fragment number (int, 1-indexed) to a sorted list of
        ``(name, sequence)`` tuples.
    """
    # read in csv
    library_df = pd.read_csv(fragment_csv)
    
    # get number of fragments
    num_fragments = len(library_df['fragment'].unique())
    
    # build fragment dictionary
    fragment_dictionary = {}
    for i in range(num_fragments):
        tmp = library_df[library_df['fragment']==i+1]
        fragment_dictionary[i+1] = [(name,frag) for name,frag in zip(tmp['name'].tolist(),tmp['seq'].tolist())]
    
    # alphabetically sort all fragments
    for x in fragment_dictionary:
        fragment_dictionary[x].sort()

    return fragment_dictionary


def make_all_sequences(fragment_dictionary, seq_list=[('','')], frag_num=1, to_join_on=':'):
    """Recursively enumerate all combinatorial sequences from a fragment dictionary.

    Builds full-length sequences by concatenating one entry from each
    fragment region. Names are joined with ``to_join_on`` (default ``:``)
    as the delimiter.

    Args:
        fragment_dictionary: Dict from :func:`get_fragment_dictionary`.
        seq_list: Accumulated list of ``(name, sequence)`` tuples (internal
            recursion state).
        frag_num: Current fragment region number (1-indexed).
        to_join_on: Delimiter for joining fragment names.

    Returns:
        List of ``(name, sequence)`` tuples covering all combinations.
    """
    new_seq_list = []
    for name,seq in seq_list:
        for name_to_add,frag_to_add in fragment_dictionary[frag_num]:
            if frag_num == 1:
                catname = name_to_add
            else:
                catname = name+to_join_on+name_to_add
            new_seq_list.append((catname,seq+frag_to_add))
    
    frag_num += 1
    if frag_num-1 < len(fragment_dictionary):
        return make_all_sequences(fragment_dictionary, seq_list=new_seq_list, frag_num=frag_num)
    else:
        print(f'Generated {len(new_seq_list)} possible sequences.')
        return new_seq_list


def get_all_sequences(fragment_dictionary):
    """Enumerate and sort all combinatorial sequences in the library.

    Args:
        fragment_dictionary: Dict from :func:`get_fragment_dictionary`.

    Returns:
        Sorted list of ``(name, sequence)`` tuples.
    """
    all_seqs = make_all_sequences(fragment_dictionary)
    all_seqs.sort()
    return all_seqs

def featurize_fragments(names, fragment_dictionary, to_split_on=':', num_fragments=None):
    """Convert compound fragment names into integer index tensors.

    Each name is split on ``to_split_on`` and each token is looked up in
    the corresponding fragment region to produce a per-region index.

    Args:
        names: List of compound names (e.g. ``"fragA:fragB:fragC"``).
        fragment_dictionary: Dict from :func:`get_fragment_dictionary`.
        to_split_on: Delimiter used in compound names.
        num_fragments: If set, truncate to this many fragments (useful when
            names carry extra suffixes).

    Returns:
        Integer tensor of shape ``(len(names), num_regions)`` with fragment
        indices suitable for embedding layers.
    """
    frag_nums = []
    for n in names:
        frag_list = n.split(to_split_on)

        if num_fragments is not None:
            frag_list = frag_list[:num_fragments]
            # sometimes the names have extra information appended to it

        # check to make sure split is correct
        assert len(frag_list) == len(fragment_dictionary), "Check you are splitting correctly"
        
        tmp_nums = []
        for i,f in enumerate(frag_list):
            frag_idx = [x[0] for x in fragment_dictionary[i+1]].index(f)
            tmp_nums.append(frag_idx)
            
        frag_nums.append(tmp_nums)

    return torch.tensor(frag_nums)
