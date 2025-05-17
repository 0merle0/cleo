#!/usr/bin/env python

import argparse
import os
import yaml
import pandas as pd
from typing import Dict, Any
from Bio.PDB import PDBParser
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure import distance
import numpy as np
import glob
import json
import copy
import pdb as pdb_lib

## load all metrics function from metrics.py
from metrics import *

## import cifutils functions ##
from cifutils.utils.io_utils import load_any


###### PDB parsing functions ######

def parse_pdb(file_path: str) -> struc.AtomArray:
    """
    Parse a PDB/CIF file and extract the first model as an AtomArray.
    This is done using the load any function from cifutils

    Parameters
    ----------
    file_path : str
        Path to the PDB file or a CIF File 

    Returns
    -------
    AtomArray
        The first model's AtomArray.
    """
    arr_s = load_any(file_path, extra_fields = ['charge'])
    atom_array = arr_s[0]
    final_arr = atom_array[atom_array.element != 'H'] # remove hydrogens
    return final_arr

def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def simplify_dict(d):
    for key, value in d.items():
        if isinstance(value, list):
            # Replace list with its first element (or None if empty)
            d[key] = value[0] if value else None
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            d[key] = simplify_dict(value)
    return d


def read_config(config_path: str) -> Dict[str, Any]:
    """Read YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_atom_names(pdb_array: struc.AtomArray, residue_spec: str) -> list:
    """
    Given a residue specifier like 'A128', return the list of atom names
    for that residue in the given pdb_array.

    Parameters
    ----------
    pdb_array : biotite.structure.AtomArray
        Parsed PDB structure.
    residue_spec : str
        String like 'A128', where the first character(s) is chain ID
        (e.g., 'A'), followed by the residue number (e.g., '128').

    Returns
    -------
    list
        A list of atom names for the specified chain and residue ID.
    """
    chain_id = residue_spec[0]
    res_id = int(residue_spec[1:])

    # Boolean mask for chain + residue
    mask = (
        (pdb_array.chain_id == chain_id) &
        (pdb_array.res_id == res_id)
    )
    # Collect unique atom names for this residue
    atom_names = np.unique(pdb_array.atom_name[mask]).tolist()

    return atom_names

def compute_metrics(dir, name):
    """
    Compute mean and variance of metrics from JSON files in a directory.

    Parameters:
        dir (str): Path to the directory containing JSON files.
        name (str): Common name pattern for JSON files (eg enz name)

    Returns:
        dict: Dictionary containing mean and variance for each metric.
    """
    # Glob all JSON files
    # pdb_lib.set_trace()
    all_jsons = glob.glob(dir + f"*/{name}*/*{name}*summary_confidences.json")

    # Initialize a dictionary to store aggregated data
    metrics = {
        'ptm': [],
        'iptm': [],
        'design_substrate_pae': [],
        'substrate_ptm': [],
        'aggregate_score': []
    }

    # Iterate over each JSON file
    json_file = all_jsons[0]
    with open(json_file, 'r') as f:
        data = json.load(f)
    # pdb_lib.set_trace()
    # General metrics
    metrics['ptm'].append(data.get('ptm', [None]))
    metrics['iptm'].append(data.get('iptm', [None]))
    metrics['aggregate_score'].append(data.get('ranking_score', [None]))
    metrics['substrate_ptm'].append(data.get('chain_ptm', [None]))
    # Check for chain-specific metrics
    pae = data.get('chain_pair_pae_min')
    # pdb_lib.set_trace()
    if pae and len(pae[0]) == 2:  # Ensure there are at least 2 chains
        metrics['design_substrate_pae'].append((pae[0][1] + pae[1][0]) / 2)
    elif pae and len(pae[0]) > 2:
        metrics['design_substrate_pae'].append((pae[0][0][1] + pae[0][1][0]) / 2)
    else:
        metrics['design_substrate_pae'].append(None)

    return metrics

def parse_pdb_stack(file_path: str) -> struc.AtomArrayStack:
    """
    Parse a PDB file and extract *all* models as an AtomArrayStack.
    This will handle multi-model PDB files (i.e., multiple states).
    If the file has only one model, the stack depth will be 1.

    Parameters
    ----------
    file_path : str
        Path to the PDB file.

    Returns
    -------
    AtomArrayStack
        A stack containing one slice per model in the PDB file.
        Hydrogens are filtered out.
    """
    arr_s = load_any(file_path, extra_fields = ['charge'])
    atom_array_stack = arr_s[..., arr_s.element != 'H']
    return atom_array_stack

def compute_ensemble_statistics(all_stats):
    """
    Calculate ensemble statistics (mean and variance) from a collection of metrics.
    
    Parameters
    ----------
    all_stats : dict
        Dictionary containing lists of metrics for an ensemble of structures
        
    Returns
    -------
    dict
        Dictionary containing mean and variance for each metric
    """
    ensemble_stats = {
        "rmsd_motif": {
            "mean": round(np.mean(all_stats["rmsd_motif"]), 4),
            "variance": round(np.var(all_stats["rmsd_motif"]), 4)
        },
        "rmsd_full_motif": {
            "mean": round(np.mean(all_stats["rmsd_full_motif"]), 4),
            "variance": round(np.var(all_stats["rmsd_full_motif"]), 4)
        },
        "ca_rmsd": {
            "mean": round(np.mean([x for x in all_stats["ca_rmsd"] if x is not None]), 4) if any(x is not None for x in all_stats["ca_rmsd"]) else None,
            "variance": round(np.var([x for x in all_stats["ca_rmsd"] if x is not None]), 4) if any(x is not None for x in all_stats["ca_rmsd"]) else None
        },
        "per_res_rmsd": {},
        "distances": {},
        "plddt": {},
        "angles": {},
        "dihedrals": {}
    }

    # Aggregate stats for individual residues
    for res, values in all_stats["per_res_rmsd"].items():
        ensemble_stats["per_res_rmsd"][res] = {
            "mean": round(np.mean(values), 4),
            "variance": round(np.var(values), 4)
        }

    # Aggregate stats for distances
    for pair, values in all_stats["distances"].items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ensemble_stats["distances"][pair] = {
                "mean": round(np.mean(valid_values), 4),
                "variance": round(np.var(valid_values), 4)
            }
        else:
            ensemble_stats["distances"][pair] = {
                "mean": None,
                "variance": None
            }

    # Aggregate stats for angles
    for angle_name, values in all_stats["angles"].items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ensemble_stats["angles"][angle_name] = {
                "mean": round(np.mean(valid_values), 4),
                "variance": round(np.var(valid_values), 4)
            }
        else:
            ensemble_stats["angles"][angle_name] = {
                "mean": None,
                "variance": None
            }

    # Aggregate stats for dihedrals
    for dihedral_name, values in all_stats["dihedrals"].items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            ensemble_stats["dihedrals"][dihedral_name] = {
                "mean": round(np.mean(valid_values), 4),
                "variance": round(np.var(valid_values), 4)
            }
        else:
            ensemble_stats["dihedrals"][dihedral_name] = {
                "mean": None,
                "variance": None
            }

    # Aggregate stats for pLDDT
    for res, scores in all_stats["plddt"].items():
        # If scores is a simple list of values (per-residue pLDDT)
        if isinstance(scores, list):
            ensemble_stats["plddt"][res] = {
                "mean": round(np.mean(scores), 4),
                "variance": round(np.var(scores), 4)
            }
            
    return ensemble_stats

def get_state_stats(enzyme_names, state_specs, datadir):
    """Process statistics for each enzyme in the given state."""
    # Parse reference structure
    # pdb_lib.set_trace()
    ref_array = parse_pdb(state_specs['ref'])
    
    # Extract residue specifications
    motif_resis = state_specs.get('motif_resis', [])
    motif_atoms_spec = state_specs.get('motif_atoms')
    motif_lig = state_specs.get('motif_lig', [])
    motif_lig_atoms_spec = state_specs.get('motif_lig_atoms')
    
    # Build motif dictionary
    motif_dict, motif_dict_full = build_motif_dictionaries(ref_array, motif_resis, motif_lig, motif_atoms_spec, motif_lig_atoms_spec)

    # Extract other specifications
    distance_pairs = state_specs.get('distances', {})
    tracked_angles = state_specs.get('tracked_angles', {})
    tracked_dihedrals = state_specs.get('tracked_dihedrals', {})
    
    # Extract chiral centers
    chiral_centers = state_specs.get('chiral_centers', {})

    # make all names lowercase
    enzyme_names = [name.lower() for name in enzyme_names]

    # Process each enzyme
    all_stats = []
    for enz_name in enzyme_names:
        try:
            stats = get_metrics(
                pred_dir=datadir,
                ref_pdb=state_specs['ref'],
                motif_dict=motif_dict,
                motif_dict_full=motif_dict_full,
                distance_pairs=distance_pairs,
                tracked_angles=tracked_angles,
                tracked_dihedrals=tracked_dihedrals,
                enz_name=enz_name,
                chiral_centers=chiral_centers
            )
            flattened_stats = flatten_dict(simplify_dict(stats))
            flattened_stats['name'] = enz_name
            all_stats.append(flattened_stats)
            
        except Exception as e:
            print(f"Warning: Skipping enzyme {enz_name} due to error: {str(e)} for {state_specs['ref']}")
            continue

    return all_stats

def map_catalytic_residues(pred_dir, enz_name, ref_array, motif_dict, motif_dict_full, distance_pairs=None, tracked_angles=None, tracked_dihedrals=None):
    """
    Map catalytic residues between input and design using compiled metrics.
    Updates all related data structures based on the mapping.
    
    Parameters
    ----------
    pred_dir : str
        Directory containing the prediction files
    enz_name : str
        Name of the enzyme
    ref_array : biotite.structure.AtomArray
        Reference structure to be modified based on mapping
    motif_dict : dict
        Dictionary containing motif information
    motif_dict_full : dict
        Full dictionary containing motif information
    distance_pairs : dict, optional
        Dictionary of distance pairs to be updated
    tracked_angles : dict, optional
        Dictionary of tracked angles to be updated
    tracked_dihedrals : dict, optional
        Dictionary of tracked dihedrals to be updated
    
    Returns
    -------
    tuple
        (ref_array, motif_dict, motif_dict_full, distance_pairs, tracked_angles, tracked_dihedrals)
    
    Raises
    ------
    ValueError
        If mapping cannot be performed
    """
    try:
        metrics_file = os.path.join(pred_dir, 'compiled_metrics.csv')
        if not os.path.exists(metrics_file):
            catres = motif_dict.keys()
            reverse_mapping = dict(zip(catres, catres))
            return ref_array, motif_dict, motif_dict_full, distance_pairs, tracked_angles, tracked_dihedrals, reverse_mapping
        metrics_df = pd.read_csv(metrics_file)
        bb_name = enz_name.rsplit('_', 1)[0]
        entry = metrics_df[metrics_df['name'].str.lower() == bb_name.lower()].iloc[0]
        catres1 = entry['catres_input'].split(',')
        catres2 = entry['catres_design'].split(',')
        mapping = dict(zip(catres1, catres2))
        reverse_mapping = dict(zip(catres2, catres1))
        # Modify reference array based on mapping
        ref_array_modified = ref_array.copy()
        
        # Available chain IDs for reassignment (excluding common ones)
        available_chains = ['X', 'Y', 'Z', 'U', 'V', 'W']
        chain_idx = 0
        
        # TODO: this is hacky, we should figure out a better way to do this
        # First pass: handle conflicts by moving atoms that might be at target positions
        for old_res_spec, new_res_spec in mapping.items():
            # Parse old and new specifications
            old_chain = old_res_spec[0]
            old_res_id = int(old_res_spec[1:])
            new_chain = new_res_spec[0]
            new_res_id = int(new_res_spec[1:])
            
            # Check if there are already atoms at the target position
            # (but not the ones we're about to move)
            target_mask = ((ref_array_modified.chain_id == new_chain) & 
                           (ref_array_modified.res_id == new_res_id) &
                           ~((ref_array_modified.chain_id == old_chain) & 
                             (ref_array_modified.res_id == old_res_id)))
            
            # If there are atoms at the target position, move them to a new chain
            if np.any(target_mask):
                # Get next available chain ID
                new_chain_id = available_chains[chain_idx]
                chain_idx = (chain_idx + 1) % len(available_chains)
                
                # Move conflicting atoms to the new chain
                ref_array_modified.chain_id[target_mask] = new_chain_id
                
                # Also update any mappings that referenced these atoms
                for key in list(mapping.keys()):
                    if key.startswith(new_chain) and int(key[1:]) == new_res_id:
                        old_value = mapping[key]
                        # Create new key with the new chain ID
                        new_key = f"{new_chain_id}{key[1:]}"
                        mapping[new_key] = old_value
                        # Remove the old key
                        del mapping[key]
        
        # Second pass: now safely update residues according to the mapping
        for old_res_spec, new_res_spec in mapping.items():
            # Parse old and new specifications
            old_chain = old_res_spec[0]
            old_res_id = int(old_res_spec[1:])
            new_chain = new_res_spec[0]
            new_res_id = int(new_res_spec[1:])
            
            # Create mask for atoms with matching chain and residue ID
            mask = (ref_array_modified.chain_id == old_chain) & (ref_array_modified.res_id == old_res_id)
            
            # Update both chain ID and residue ID
            ref_array_modified.chain_id[mask] = new_chain
            ref_array_modified.res_id[mask] = new_res_id

        # Update motif dictionaries
        mapped_motif_dict = {mapping.get(old_key, old_key): value 
                            for old_key, value in motif_dict.items()}
        mapped_motif_dict_full = {mapping.get(old_key, old_key): value 
                                 for old_key, value in motif_dict_full.items()}
        
        # Update distance_pairs if provided
        if distance_pairs:
            updated_distance_pairs = {}
            for key, spec in distance_pairs.items():
                new_atoms = []
                for atom_pair in spec['atoms']:
                    chain_res, atom_name = atom_pair
                    # Apply mapping if chain+residue exists in mapping
                    new_chain_res = mapping.get(chain_res, chain_res)
                    new_atoms.append([new_chain_res, atom_name])
                
                # Create new entry with updated atoms
                updated_spec = spec.copy()
                updated_spec['atoms'] = new_atoms
                updated_distance_pairs[key] = updated_spec
            distance_pairs = updated_distance_pairs
        
        # Update tracked_angles if provided
        if tracked_angles:
            updated_tracked_angles = {}
            for key, spec in tracked_angles.items():
                new_atoms = []
                for atom_spec in spec['atoms']:
                    chain_res, atom_name = atom_spec
                    # Apply mapping if chain+residue exists in mapping
                    new_chain_res = mapping.get(chain_res, chain_res)
                    new_atoms.append([new_chain_res, atom_name])
                
                # Create new entry with updated atoms
                updated_spec = spec.copy()
                updated_spec['atoms'] = new_atoms
                updated_tracked_angles[key] = updated_spec
            tracked_angles = updated_tracked_angles
        
        # Update tracked_dihedrals if provided
        if tracked_dihedrals:
            updated_tracked_dihedrals = {}
            for key, spec in tracked_dihedrals.items():
                new_atoms = []
                for atom_spec in spec['atoms']:
                    chain_res, atom_name = atom_spec
                    # Apply mapping if chain+residue exists in mapping
                    new_chain_res = mapping.get(chain_res, chain_res)
                    new_atoms.append([new_chain_res, atom_name])
                
                # Create new entry with updated atoms
                updated_spec = spec.copy()
                updated_spec['atoms'] = new_atoms
                updated_tracked_dihedrals[key] = updated_spec
            tracked_dihedrals = updated_tracked_dihedrals
        
        print(f"Mapped {len(reverse_mapping)} residues")
        print(f"forward mapping: {mapping}")
        print(f"updated distance pairs: {distance_pairs}")
        print(f"updated tracked angles: {tracked_angles}")
        print(f"updated tracked dihedrals: {tracked_dihedrals}")
        return ref_array_modified, mapped_motif_dict, mapped_motif_dict_full, distance_pairs, tracked_angles, tracked_dihedrals, reverse_mapping
    except Exception as e:
        raise ValueError(f"Could not map contigs: {str(e)}")

def build_motif_dictionaries(ref_array, motif_resis, motif_lig, motif_atoms_spec=None, motif_lig_atoms_spec=None):
    """
    Build dictionaries of motif atoms for residues and ligands.
    
    Parameters
    ----------
    ref_array : biotite.structure.AtomArray
        Reference structure containing the atoms
    motif_resis : list
        List of residue identifiers for the motif
    motif_lig : list
        List of ligand identifiers for the motif
    motif_atoms_spec : dict, optional
        Specification of atoms to include for each residue
    motif_lig_atoms_spec : dict, optional
        Specification of atoms to include for each ligand
    
    Returns
    -------
    tuple
        (motif_dict, motif_dict_full) - dictionaries mapping residue IDs to atom names
    """
    # Build motif dictionary
    motif_dict = {}
    if motif_atoms_spec is None:
        for res in motif_resis:
            atoms = [atom for atom in get_atom_names(ref_array, res) if 'H' not in atom]
            motif_dict[res] = atoms
    else:
        for res in motif_resis:
            if res in motif_atoms_spec:
                motif_dict[res] = motif_atoms_spec[res]
            else:
                atoms = [atom for atom in get_atom_names(ref_array, res) if 'H' not in atom]
                motif_dict[res] = atoms

    # Build full motif dictionary including ligand atoms
    motif_dict_full = copy.deepcopy(motif_dict)
    motif_lig_atoms = {}

    if motif_lig_atoms_spec is None:
        for res in motif_lig:
            atoms = [atom for atom in get_atom_names(ref_array, res) if 'H' not in atom]
            motif_lig_atoms[res] = atoms
    else:
        for res in motif_lig:
            if res in motif_lig_atoms_spec:
                motif_lig_atoms[res] = motif_lig_atoms_spec[res]
            else:
                atoms = [atom for atom in get_atom_names(ref_array, res) if 'H' not in atom]
                motif_lig_atoms[res] = atoms

    for key, val in motif_lig_atoms.items():
        if key not in motif_dict_full:
            motif_dict_full[key] = []
        motif_dict_full[key].extend(val)
        motif_dict_full[key] = list(set(motif_dict_full[key]))
    
    return motif_dict, motif_dict_full

def process_pdb_file(pdb_file, ref_array, motif_dict, motif_dict_full, distance_pairs, 
                    tracked_angles=None, tracked_dihedrals=None, enz_name="", reverse_mapping=None):
    """
    Process a single PDB file to calculate various structural metrics.
    
    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to be processed
    ref_array : biotite.structure.AtomArray
        Reference structure for alignment and comparison
    motif_dict : dict
        Dictionary mapping residue IDs to atom names (motif only)
    motif_dict_full : dict
        Dictionary mapping residue IDs to atom names (motif + ligand)
    distance_pairs : dict
        Dictionary specifying distance pairs to calculate
    tracked_angles : dict, optional
        Dictionary specifying angles to track
    tracked_dihedrals : dict, optional
        Dictionary specifying dihedrals to track
    enz_name : str, optional
        Name of the enzyme for error reporting
        
    Returns
    -------
    dict
        Dictionary containing all calculated metrics for this PDB
    """
    # Initialize stats dictionary
    stats = {
        "rmsd_motif": [],
        "rmsd_full_motif": [],
        "ca_rmsd": [],
        "per_res_rmsd": {},
        "distances": {},
        "angles": {},
        "dihedrals": {},
        "plddt": {}
    }
    
    # Parse the PDB file
    # mob_array_stack = parse_pdb_stack(pdb_file)
    arr_s = load_any(pdb_file, extra_fields = ['charge', 'b_factor'])
    mob_array_stack = arr_s[..., arr_s.element != 'H']

    # Process each model in the stack
    for model_idx in range(mob_array_stack.stack_depth()):
        mob_array = mob_array_stack[model_idx]
        
        ##########
        # hacky fix for AF3 giving res_id=-1
        # TODO: fix this properly
        res_ids = mob_array.res_id
        mask = (mob_array.chain_id == 'B') | (mob_array.chain_id == 'C')
        res_ids[mask] = [1]*len(res_ids[mask])
        mob_array.res_id = res_ids
        ##########

        # Alignment and RMSD calculations
        _, ca_rmsd = align_by_ca(ref_array, mob_array)
        aligned_mob_array = align_by_residue_dict(ref_array, mob_array, motif_dict)
        
        # Compute RMSD for motif
        rmsd_motif = compute_rmsd_by_residue_dict(ref_array, aligned_mob_array, motif_dict)
        stats["rmsd_motif"].append(rmsd_motif)
        stats["ca_rmsd"].append(ca_rmsd)

        # Compute RMSD for full motif
        rmsd_full_motif = compute_rmsd_by_residue_dict(ref_array, aligned_mob_array, motif_dict_full)
        stats["rmsd_full_motif"].append(rmsd_full_motif)

        # Per-residue RMSD
        for res, atoms in motif_dict_full.items():
            res_name = reverse_mapping.get(res, res)
            temp_dict = {res: atoms}
            rmsd_val = compute_rmsd_by_residue_dict(ref_array, aligned_mob_array, temp_dict)
            if res_name not in stats["per_res_rmsd"]:
                stats["per_res_rmsd"][res_name] = []
            stats["per_res_rmsd"][res_name].append(rmsd_val)

        # Distances
        if distance_pairs:
            for pair_name, pair_spec in distance_pairs.items():
                if pair_name not in stats["distances"]:
                    stats["distances"][pair_name] = []
                try:
                    distance = calculate_distance(aligned_mob_array, pair_spec["atoms"])
                    stats["distances"][pair_name].append(distance)
                except ValueError as e:
                    print(f"Warning: Could not calculate distance {pair_name} for {enz_name}: {str(e)}")
                    stats["distances"][pair_name].append(None)

        # Calculate angles if specified
        if tracked_angles:
            for angle_name, angle_spec in tracked_angles.items():
                if angle_name not in stats["angles"]:
                    stats["angles"][angle_name] = []
                try:
                    angle = calculate_angle(aligned_mob_array, angle_spec["atoms"])
                    stats["angles"][angle_name].append(angle)
                except ValueError as e:
                    print(f"Warning: Could not calculate angle {angle_name} for {enz_name}: {str(e)}")
                    stats["angles"][angle_name].append(None)

        # Calculate dihedrals if specified
        if tracked_dihedrals:
            for dihedral_name, dihedral_spec in tracked_dihedrals.items():
                if dihedral_name not in stats["dihedrals"]:
                    stats["dihedrals"][dihedral_name] = []
                try:
                    dihedral = calculate_dihedral(aligned_mob_array, dihedral_spec["atoms"])
                    stats["dihedrals"][dihedral_name].append(dihedral)
                except ValueError as e:
                    print(f"Warning: Could not calculate dihedral {dihedral_name} for {enz_name}: {str(e)}")
                    stats["dihedrals"][dihedral_name].append(None)

        # Extract pLDDT values
        plddt_dict = extract_b_factors_by_residue(mob_array, motif_dict_full)
        for res, avg_b_factor in plddt_dict.items():
            res_name = reverse_mapping.get(res, res)
            if res_name not in stats["plddt"]:
                stats["plddt"][res_name] = []
            stats["plddt"][res_name].append(avg_b_factor)
            
    return stats

def get_metrics(pred_dir, ref_pdb, motif_dict, motif_dict_full, distance_pairs, enz_name, tracked_angles=None, tracked_dihedrals=None, chiral_centers=None):
    """
    Compute RMSD values and geometric measurements for an ensemble of PDB files.
    """
    # pdb_lib.set_trace()
    ref_array = parse_pdb(ref_pdb)
    pdb_best = glob.glob(pred_dir + f"/{enz_name}/*{enz_name}*.cif")[0]
    all_pdbs = glob.glob(pred_dir + f"/{enz_name}/*/*.cif")
    all_stats = {
        "rmsd_motif": [],
        "rmsd_full_motif": [],
        "ca_rmsd": [],
        "per_res_rmsd": {},
        "distances": {},
        "plddt": {},
        "angles": {},
        "dihedrals": {}
    }
    best_stats = {}
    for key, value in all_stats.items():
        best_stats[key + "_best"] = value

    # Map catalytic residues and modify all arguments
    _ref_array, _motif_dict, _motif_dict_full, _distance_pairs, _tracked_angles, _tracked_dihedrals, reverse_mapping = map_catalytic_residues(pred_dir, enz_name, ref_array, motif_dict, motif_dict_full, distance_pairs, tracked_angles, tracked_dihedrals)

    stats = process_pdb_file(pdb_best, _ref_array, _motif_dict, _motif_dict_full, _distance_pairs, 
                            tracked_angles=_tracked_angles, tracked_dihedrals=_tracked_dihedrals, 
                            enz_name=enz_name, reverse_mapping=reverse_mapping)
    for key, value in stats.items():
        best_stats[key + "_best"] = value

    # Analyze chirality separately for the best PDB file
    if chiral_centers:
        chirality_results = analyze_chirality(parse_pdb(pdb_best), chiral_centers)
        best_stats["chirality_best"] = chirality_results

    for pdb_file in all_pdbs:
        # Process current PDB file
        curr_stats = process_pdb_file(pdb_file, _ref_array, _motif_dict, _motif_dict_full, _distance_pairs, 
                            tracked_angles=_tracked_angles, tracked_dihedrals=_tracked_dihedrals, 
                            enz_name=enz_name, reverse_mapping=reverse_mapping)
        
        # Aggregate simple list stats
        for key in ["rmsd_motif", "rmsd_full_motif", "ca_rmsd"]:
            all_stats[key].extend(curr_stats[key])
        
        # Aggregate nested dictionary stats
        for nested_key in ["per_res_rmsd", "distances", "plddt", "angles", "dihedrals"]:
            for k, v in curr_stats[nested_key].items():
                if k not in all_stats[nested_key]:
                    all_stats[nested_key][k] = []
                all_stats[nested_key][k].extend(v)

    # Calculate ensemble statistics using the new function
    ensemble_stats = compute_ensemble_statistics(all_stats)

        
    agg_metrics = compute_metrics(pred_dir, enz_name)
    ensemble_stats.update(agg_metrics)

    # Include chirality stats in ensemble_stats from the best PDB
    if "chirality_best" in best_stats:
        ensemble_stats["chirality"] = best_stats["chirality_best"]

    # merge best_stats and ensemble_stats
    merged_stats = {**best_stats, **ensemble_stats}
    
    return merged_stats

def main():
    parser = argparse.ArgumentParser(description='Analyze CHAI metrics for enzyme designs')
    parser.add_argument('--input_list', type=str, required=True,
                       help='File containing list of enzymes to analyze')
    parser.add_argument('--outcsv', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--specs_json', type=str, required=True,
                       help='Path to specs JSON file')
    parser.add_argument('--datadir', type=str, default=None,
                       help='Optional: Base directory for prediction outputs')
    
    args = parser.parse_args()

    # Read specs json
    with open(args.specs_json, 'r') as f:
        specs = json.load(f)

    states = list(specs.keys())
    # Read list of enzymes to analyze
    with open(args.input_list, 'r') as f:
        enzyme_names = [line.strip() for line in f]

    dataframes = []

    for state in states:
        # Determine the output directory
        if args.datadir is None:
            # Use outdir from specs if datadir not provided
            pred_dir = specs[state].get('out_dir')
            if pred_dir is None:
                raise ValueError(f"No datadir provided and no outdir found in specs for state {state}")
        else:
            # Construct path using datadir and state name
            pred_dir = args.datadir
        
        enz_stats = get_state_stats(enzyme_names, specs[state], pred_dir)
        df = pd.DataFrame(enz_stats)
        df = df.rename(columns=lambda x: f"{state}_{x}" if x != 'name' else x)
        df.set_index('name', inplace=True)
        dataframes.append(df)

    # Concatenate all DataFrames horizontally (i.e., merge on 'name')
    df_stats = pd.concat(dataframes, axis=1)
    df_stats.reset_index(inplace=True)
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    # Save results
    df_stats.to_csv(args.outcsv, index=False)


if __name__ == "__main__":
    main()