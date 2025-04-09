#!/usr/bin/env python


import pandas as pd
from typing import Dict, Any
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure import distance
import numpy as np
import glob
import json
import copy
import pdb
from rdkit import Chem
from rdkit.Chem import AllChem
from cifutils.tools.rdkit import atom_array_to_rdkit
import sys

def find_atom_index(atom_array, chain_id, residue_number, atom_name):
    """
    Find the index of an atom in an AtomArray based on chain ID, residue number, and atom name.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        The AtomArray to search.
    chain_id : str
        The chain ID of the atom.
    residue_number : int
        The residue number of the atom.
    atom_name : str
        The name of the atom.
    """
    mask = (
        (atom_array.chain_id == chain_id) &
        (atom_array.res_id == residue_number) &
        (atom_array.atom_name == atom_name)
    )
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(f"Atom {atom_name} in residue {chain_id}{residue_number} not found.")
    elif len(indices) > 1:
        raise ValueError(f"Multiple atoms named {atom_name} in residue {chain_id}{residue_number} found.")
    return indices[0]

def get_atoms_from_specs(atom_array, atom_specs):
    """
    Extract atoms from an atom array based on chain and residue specifications.
    
    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Structure containing the atoms.
    atom_specs : list
        List of [chain_res, atom_name] pairs, where chain_res is a string
        containing chain ID and residue number (e.g., 'A123').
    
    Returns
    -------
    list
        List of atoms extracted from the atom array.
    """
    atoms = []
    for chain_res, atom_name in atom_specs:
        chain_id = chain_res[0]
        res_num = int(chain_res[1:])
        idx = find_atom_index(atom_array, chain_id, res_num, atom_name)
        atoms.append(atom_array[idx])
    return atoms

def calculate_dihedral(atom_array, atom_specs):
    """
    Calculate dihedral angle between four atoms using Biotite.
    """
    atoms = get_atoms_from_specs(atom_array, atom_specs)
    dihedral_rad = struc.dihedral(atoms[0], atoms[1], atoms[2], atoms[3])
    return np.degrees(dihedral_rad)

def calculate_angle(atom_array, atom_specs):
    """
    Calculate angle between three atoms using Biotite.
    """
    atoms = get_atoms_from_specs(atom_array, atom_specs)
    angle_rad = struc.angle(atoms[0], atoms[1], atoms[2])
    return np.degrees(angle_rad)

def calculate_distance(atom_array, atom_specs):
    """
    Calculate distance between two atoms using Biotite.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Structure containing the atoms.
    atom_specs : list
        List of two [chain_res, atom_name] pairs defining the atoms.
        its a list of lists

    Returns
    -------
    float
        Distance in Angstroms.
    """
    atoms = get_atoms_from_specs(atom_array, atom_specs)
    # Get the coordinates of the two atoms
    coord1 = atoms[0].coord
    coord2 = atoms[1].coord
    # Calculate Euclidean distance
    return np.linalg.norm(coord1 - coord2) 

def align_by_residue_dict(
    ref_array: struc.AtomArray,
    mob_array: struc.AtomArray,
    residue_dict: dict
    ) -> struc.AtomArray:
    """
    Align two PDB structures (ref_array and mob_array) based on specific
    residues and atom names provided in residue_dict. The keys are strings
    like 'A128' and the values are lists of atom names like ['NE', 'CD', 'CZ'].

    Parameters
    ----------
    ref_array : biotite.structure.AtomArray
        Reference structure.
    mob_array : biotite.structure.AtomArray
        Mobile (target) structure.
    residue_dict : dict
        Dictionary mapping residue specifiers to lists of atom names.
        For example:
            {
             'A128': ['NE', 'CD', 'CZ'],
             'A95': ['OD1', 'CG'],
             'A53': ['NH2', 'CZ'],
             'A71': ['NE2', 'CD2', 'CE1']
            }

    Returns
    -------
    aligned_mob_array : biotite.structure.AtomArray
        A new AtomArray for the mobile structure, aligned to ref_array
        by the specified atoms.
    """
    # Collect the coordinates in the same order
    coords_ref = []
    coords_mob = []
    
    # For tracking atom ordering
    matched_atoms = []
    
    for residue_spec, atom_list in residue_dict.items():
        chain_id = residue_spec[0]
        res_id   = int(residue_spec[1:])
        
        # Boolean masks to select only the specified chain, residue, and atoms
        mask_ref = (
            (ref_array.chain_id == chain_id) &
            (ref_array.res_id == res_id) &
            np.isin(ref_array.atom_name, atom_list)
        )
        mask_mob = (
            (mob_array.chain_id == chain_id) &
            (mob_array.res_id == res_id) &
            np.isin(mob_array.atom_name, atom_list)
        )
        
        # For safety: check that both have the same number of atoms
        num_ref = mask_ref.sum()
        num_mob = mask_mob.sum()
        if num_ref != num_mob:
            raise ValueError(
                f"Mismatch in number of atoms for residue {residue_spec} "
                f"between ref and mob arrays. Ref: {num_ref}, Mob: {num_mob}"
            )
        elif num_ref == 0:
            raise ValueError(
                f"No matching atoms found for residue {residue_spec} in ref/mob arrays."
            )
        
        # Extract atom indices to preserve order
        ref_indices = np.where(mask_ref)[0]
        mob_indices = np.where(mask_mob)[0]
        
        # Sort atoms within residue by atom name to ensure consistent order
        sorted_ref_indices = ref_indices[np.argsort(ref_array.atom_name[ref_indices])]
        sorted_mob_indices = mob_indices[np.argsort(mob_array.atom_name[mob_indices])]
        
        # Append coordinates
        coords_ref.append(ref_array.coord[sorted_ref_indices])
        coords_mob.append(mob_array.coord[sorted_mob_indices])
        
        # Track matched atoms
        matched_atoms.extend(ref_array.atom_name[sorted_ref_indices].tolist())
    
    # Concatenate into Nx3 arrays
    coords_ref = np.vstack(coords_ref)
    coords_mob = np.vstack(coords_mob)
    
    # Superimpose and get the transformation
    fitted_mob, transform = struc.superimpose(coords_ref, coords_mob)
    aligned_mob_array = transform.apply(mob_array)
    
    return aligned_mob_array

def align_by_ca(
    ref_array: struc.AtomArray,
    mob_array: struc.AtomArray
    ) -> tuple[struc.AtomArray, float]:
    """
    Align two PDB structures (ref_array and mob_array) based on all C-alpha (CA) atoms,
    and calculate RMSD for the aligned C-alpha atoms.

    Parameters
    ----------
    ref_array : biotite.structure.AtomArray
        Reference structure.
    mob_array : biotite.structure.AtomArray
        Mobile (target) structure.

    Returns
    -------
    aligned_mob_array : biotite.AtomArray or None
        A new AtomArray for the mobile structure, aligned to ref_array
        by C-alpha atoms. Returns None if alignment fails.
    ca_rmsd : float or None
        RMSD between the aligned C-alpha atoms of ref_array and mob_array.
        Returns None if alignment fails.
    """
    # Boolean masks for CA atoms
    ca_mask_ref = (ref_array.atom_name == "CA")
    ca_mask_mob = (mob_array.atom_name == "CA")
    
    # Coordinates
    coords_ref = ref_array.coord[ca_mask_ref]
    coords_mob = mob_array.coord[ca_mask_mob]
    
    # Check for CA atoms and handle mismatches
    if len(coords_ref) != len(coords_mob):
        return None, None
    elif len(coords_ref) == 0:
        return None, None
    
    # Superimpose and get the transformation
    fitted_mob, transform = struc.superimpose(coords_ref, coords_mob)
    aligned_mob_array = transform.apply(mob_array)
    ca_rmsd = struc.rmsd(coords_ref, fitted_mob)
    
    return aligned_mob_array, ca_rmsd

def compute_rmsd_by_residue_dict(
    ref_array: struc.AtomArray,
    mob_array: struc.AtomArray,
    residue_dict: dict
    ) -> float:
    """
    Compute the RMSD between two structures for the atoms specified in residue_dict.
    The assumption is that the reference and the mobile arrays have the same
    residue/atom naming conventions.

    Parameters
    ----------
    ref_array : biotite.structure.AtomArray
        Reference structure.
    mob_array : biotite.structure.AtomArray
        Mobile (target) structure.
    residue_dict : dict
        Dictionary mapping residue specifiers to lists of atom names.
        For example:
            {
             'A128': ['NE', 'CD', 'CZ'],
             'A95': ['OD1', 'CG'],
             'A53': ['NH2', 'CZ'],
             'A71': ['NE2', 'CD2', 'CE1']
            }

    Returns
    -------
    float
        The RMSD value for the specified atoms.
    """
    coords_ref = []
    coords_mob = []
    
    for residue_spec, atom_list in residue_dict.items():
        chain_id = residue_spec[0]
        res_id   = int(residue_spec[1:])
        
        # Boolean masks
        mask_ref = (
            (ref_array.chain_id == chain_id) &
            (ref_array.res_id == res_id) &
            np.isin(ref_array.atom_name, atom_list)
        )
        mask_mob = (
            (mob_array.chain_id == chain_id) &
            (mob_array.res_id == res_id) &
            np.isin(mob_array.atom_name, atom_list)
        )
        
        # For safety: check matching number of atoms
        num_ref = mask_ref.sum()
        num_mob = mask_mob.sum()
        if num_ref != num_mob:
            raise ValueError(
                f"Mismatch in number of atoms for residue {residue_spec} "
                f"between ref and mob arrays. Ref: {num_ref}, Mob: {num_mob}"
            )
        elif num_ref == 0:
            raise ValueError(
                f"No matching atoms found for residue {residue_spec} in ref/mob arrays."
            )
        
        # Extract atom indices to preserve order
        ref_indices = np.where(mask_ref)[0]
        mob_indices = np.where(mask_mob)[0]
        
        # Sort atoms within residue by atom name to ensure consistent order
        sorted_ref_indices = ref_indices[np.argsort(ref_array.atom_name[ref_indices])]
        sorted_mob_indices = mob_indices[np.argsort(mob_array.atom_name[mob_indices])]
        
        # Append coordinates
        coords_ref.append(ref_array.coord[sorted_ref_indices])
        coords_mob.append(mob_array.coord[sorted_mob_indices])
    
    coords_ref = np.vstack(coords_ref)
    coords_mob = np.vstack(coords_mob)
    
    # Calculate RMSD
    rmsd_value = struc.rmsd(coords_ref, coords_mob)
    return rmsd_value

def extract_b_factors_by_residue(atom_array, residue_dict):
    """
    Extract B-factors for specified residues and atoms from an atom array.
    Relevant for where metrics like confidence scores are stored in the B-factor column.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        The structure containing the atoms.
    residue_dict : dict
        Dictionary mapping residue specifiers to lists of atom names.
        For example: {'A128': ['NE', 'CD', 'CZ']}

    Returns
    -------
    dict
        A dictionary where each residue ID is a key, and the value is the average B-factor
        for the specified atoms in that residue.
    """
    b_factor_dict = {}
    for residue_spec, atom_list in residue_dict.items():
        chain_id = residue_spec[0]
        res_id = int(residue_spec[1:])
        mask = (
            (atom_array.chain_id == chain_id) &
            (atom_array.res_id == res_id) &
            np.isin(atom_array.atom_name, atom_list)
        )
        atoms = atom_array[mask]
        
        if len(atoms) > 0:
            avg_b_factor = np.mean(atoms.b_factor)
            b_factor_dict[residue_spec] = avg_b_factor
        else:
            # Handle case where no matching atoms are found
            print(f"Warning: No matching atoms found for residue {residue_spec}")
    
    return b_factor_dict

def calculate_chirality(atom_array, mol, confId, chiral_atom_name):
    """
    Calculate the CIP stereochemistry (R/S) of a given atom in a Biotite AtomArray
    using RDKit. Returns "R", "S", or None if chirality is not assigned.
    
    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        The atomic structure containing the chiral atom
    mol : rdkit.Chem.Mol
        RDKit molecule object
    confId : int
        Conformer ID in the RDKit molecule
    chiral_atom_name : str
        Name of the atom whose chirality should be calculated
        
    Returns
    -------
    str or None
        "R", "S", or None if chirality couldn't be determined
    """
    try:
        if isinstance(atom_array, struc.AtomArrayStack):
            atom_array = atom_array[0]

        # Create an editable RDKit molecule and map Biotite indices to RDKit indices
        rw_mol = Chem.RWMol()
        biotite_to_rdkit = {}
        pt = Chem.GetPeriodicTable()

        for i, atom in enumerate(atom_array):
            # Ensure element symbol is properly capitalized (e.g. 'ca' -> 'Ca')
            elem = atom.element.strip().capitalize()
            atomic_num = pt.GetAtomicNumber(elem)
            rd_idx = rw_mol.AddAtom(Chem.Atom(atomic_num))
            biotite_to_rdkit[i] = rd_idx

        # Add bonds from the atom array's bond list if available
        if atom_array.bonds is not None:
            for start, end, bond_type in atom_array.bonds.as_array():
                if start < end:  # add each bond only once
                    rw_mol.AddBond(biotite_to_rdkit[start],
                                biotite_to_rdkit[end],
                                Chem.BondType.SINGLE)  # Using SINGLE as default

        # Assign stereochemistry based on the 3D coordinates
        AllChem.AssignStereochemistryFrom3D(mol, confId=confId)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Find the target atom by its name
        target_indices = np.where(atom_array.atom_name == chiral_atom_name)[0]
        if len(target_indices) == 0:
            return None  # Silently return None instead of raising error
        
        target_index = target_indices[0]
        rd_target_idx = biotite_to_rdkit[target_index]
        rd_atom = mol.GetAtomWithIdx(rd_target_idx)

        # Return the CIP code if assigned (e.g. "R" or "S")
        if rd_atom.HasProp('_CIPCode'):
            return rd_atom.GetProp('_CIPCode')
        else:
            return None
    
    except:
        # Silently catch all exceptions and return None
        return None

def infer_charges_and_bonds_from_coord(array, ph=7.0, refine_with_rdkit=False):
    """
    Infers atomic charges and molecular bonds from 3D coordinates using OpenBabel and optionally RDKit.
    
    Parameters
    ----------
    array : biotite.structure.AtomArray
        Input structure containing atomic elements and 3D coordinates
    ph : float | None
        pH value for charge state prediction. If None, skips pH correction. Defaults to 7.0
    refine_with_rdkit : bool
        Whether to use RDKit's bond perception to refine OpenBabel results. Defaults to False
        
    Returns
    -------
    tuple
        charges (np.ndarray): Array of atomic charges
        bonds (struc.BondList): Bond list containing all detected bonds and their types
    """
    sys.path.append('/home/ssalike/git/datahub_latest/src')
    from datahub.transforms.openbabel_utils import atom_array_from_openbabel
    from openbabel import openbabel
    from cifutils.tools.rdkit import atom_array_to_rdkit, atom_array_from_rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    
    xyz_block = f"{len(array)}\n\n"
    for i in range(len(array)):
        xyz_block += f"{array.element[i]} {array.coord[i,0]:.6f} {array.coord[i,1]:.6f} {array.coord[i,2]:.6f}\n"

    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("xyz")
    obmol = openbabel.OBMol()
    obConversion.ReadString(obmol, xyz_block)
    if ph is not None:
        obmol.CorrectForPH(float(ph))
    obmol.SetAromaticPerceived(False)
    obmol.FindSSSR()
    obmol.FindLSSR()
    
    atom_array = atom_array_from_openbabel(obmol)
    # Re-ingest with RDKit to infer occasionally missing aromaticity that OpenBabel misses
    rdmol = atom_array_to_rdkit(atom_array, set_coord=True, hydrogen_policy="infer" if refine_with_rdkit else "keep")
    # Optionally use rdkit to infer bonds 
    if refine_with_rdkit:
        rdDetermineBonds.DetermineBonds(rdmol, useHueckel=True)
    atom_array = atom_array_from_rdkit(rdmol, remove_hydrogens=True, remove_inferred_atoms=True)
    return atom_array.charge, atom_array.bonds

def analyze_chirality(atom_array, chiral_centers):
    """
    Analyze chirality of atoms in chain B of a PDB file.
    
    Parameters
    ----------
    pdb_file : str
        Path to the PDB/CIF file
    chiral_centers : dict
        Dictionary mapping atom names to expected chirality (e.g., {"C1": "R", "C2": "S"})
        
    Returns
    -------
    dict
        Dictionary containing computed chirality for each center and percentage correct
    """
    chirality_results = {}
    
    try:
        # Extract chain B (ligand)
        chain_b = atom_array[atom_array.chain_id == 'B']
        
        if len(chain_b) == 0:
            # No chain B found
            for atom_name in chiral_centers:
                chirality_results[atom_name] = float('nan')
            chirality_results["correct_percentage"] = float('nan')
            return chirality_results
        
        # Infer charges and bonds
        charges, bonds = infer_charges_and_bonds_from_coord(chain_b, refine_with_rdkit=True, ph=7.0)
        ligand_fixed = chain_b.copy()
        ligand_fixed.bonds = bonds
        
        # Convert to RDKit molecule
        mol = atom_array_to_rdkit(ligand_fixed, set_coord=True, hydrogen_policy="infer", annotations_to_keep=['charge'])
        
        # Calculate chirality for each chiral center
        correct_count = 0
        total_centers = len(chiral_centers)
        
        for atom_name, expected_chirality in chiral_centers.items():
            actual_chirality = calculate_chirality(ligand_fixed, mol, 0, atom_name)
            # Store the actual chirality
            chirality_results[atom_name] = actual_chirality
            
            # Count if correct
            if actual_chirality == expected_chirality:
                correct_count += 1
        
        # Calculate percentage correct if we have valid centers
        if total_centers > 0:
            chirality_results["correct_percentage"] = (correct_count / total_centers) * 100
        else:
            chirality_results["correct_percentage"] = float('nan')
            
    except Exception as e:
        print(f"Warning: Could not calculate chirality: {str(e)}")
        # Set all chirality values to NaN
        if chiral_centers:
            for atom_name in chiral_centers:
                chirality_results[atom_name] = float('nan')
            chirality_results["correct_percentage"] = float('nan')
    
    return chirality_results