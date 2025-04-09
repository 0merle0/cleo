#!/usr/bin/env python3
"""
Script to process FASTA files to JSON files for AlphaFold3 prediction.

This script converts protein sequences from FASTA files to JSON format,
optionally adding ligands and specifying bonds between molecules.
"""

import argparse
import os
import sys
import glob
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import pdb
import numpy as np
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert FASTA files to JSON files for AlphaFold3")
    parser.add_argument("--output_dir", required=True, help="Directory to write JSON files")
    parser.add_argument("--input_csv", required=True, help="CSV file with design_name, protein_sequence and optionally catres_design, catres_input columns")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for FASTA files")
    parser.add_argument("--num_files", type=int, default=-1, help="Number of files to process")
    parser.add_argument("--ligand_data", type=str, default=None, help="JSON string with ligand information")
    parser.add_argument("--bonds", type=str, default=None, help="JSON string with bond information")
    parser.add_argument("--ptm", type=str, default=None, help="JSON string with PTM information")
    
    return parser.parse_args()

def read_sequence_from_fasta(fasta_path: str) -> str:
    """
    Read protein sequence from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Returns:
        Protein sequence as string
    """
    sequence = ""
    with open(fasta_path, 'r') as src:
        for line in src:
            line = line.strip()
            if not line.startswith('>'):
                sequence += line
    
    if not sequence:
        logger.error(f"Error: No sequence found in {fasta_path}")
        sys.exit(1)
        
    return sequence

def create_residue_mapping(catres_input, catres_design):
    """
    Create a mapping dictionary from input residue IDs to design residue IDs
    
    Args:
        catres_input: Comma-separated string of input residue IDs
        catres_design: Comma-separated string of design residue IDs
        
    Returns:
        Dictionary mapping input residue IDs to design residue IDs
    """
    input_residues = catres_input.split(',')
    design_residues = catres_design.split(',')
    
    mapping = dict(zip(input_residues, design_residues))
    logger.info(f"Residue mapping: {mapping}")
    
    return mapping

def map_bonds_with_csv_mapping(bonds: List[List[Any]], residue_mapping: Dict[str, str]) -> List[List[Any]]:
    """
    Map bond atom specifiers using the mapping from compiled_metrics.csv.
    
    Args:
        bonds: List of bond triplets in the format [["A", 142, "SG"], ["B", 1, "C8"]]
        residue_mapping: Mapping from input residue IDs to hallucinated IDs
        
    Returns:
        Updated bond list with mapped residue indices
    """
    if not bonds or not residue_mapping:
        return bonds
    
    mapped_bonds = []
    for bond_pair in bonds:
        # Only map the first entry in each bond pair (protein side)
        chain, res_num, atom_name = bond_pair[0]
        
        # Check if specifier is from protein chain
        if chain == "A":
            # Construct residue ID as it would appear in the mapping
            res_id = f"{chain}{res_num}"
            
            # Look up the residue ID in the mapping
            if res_id in residue_mapping:
                # Get hallucinated residue ID
                hal_res_id = residue_mapping[res_id]
                
                # Extract hallucinated chain and number
                hal_chain = hal_res_id[0]  # First character is chain
                hal_num = int(hal_res_id[1:])   # Rest is residue number
                
                # Create new bond pair with mapped residue
                mapped_bond = [[hal_chain, hal_num, atom_name], bond_pair[1]]
                mapped_bonds.append(mapped_bond)
            else:
                # No mapping found, keep original
                logger.warning(f"No mapping found for residue {res_id}. Using original.")
                mapped_bonds.append(bond_pair)
        else:
            # Not a protein chain, no mapping needed
            mapped_bonds.append(bond_pair)
    
    return mapped_bonds

def parse_bonds(bonds_json: str, row: pd.Series) -> List[List[Any]]:
    """
    Parse bonds JSON string into a list of bond lists.
    
    Args:
        bonds_json: JSON string representation of bonds
        fasta_name: Name of the FASTA file (used for CSV mapping)
        metrics_csv: Path to compiled_metrics.csv for residue mapping
        
    Returns:
        List of bond lists
    """
    if not bonds_json:
        return None
    
    try:
        # Parse JSON string to list
        bonds = json.loads(bonds_json)
        
        # Map bonds to hallucinated indices if CSV is provided
        if row['catres_design'] and row['catres_input']:
            residue_mapping = create_residue_mapping(row['catres_input'], row['catres_design'])
            bonds = map_bonds_with_csv_mapping(bonds, residue_mapping)
        
        return bonds
    except Exception as e:
        logger.error(f"Error parsing bonds: {e}")
        return None

def parse_ptm(ptm_json: str, residue_mapping: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Parse PTM JSON string and map residue index if needed.
    
    Args:
        ptm_json: JSON string representation of PTM data
        residue_mapping: Optional dictionary mapping input residue IDs to design residue IDs
        
    Returns:
        Dictionary with PTM information
    """
    if not ptm_json:
        return None
    
    try:
        # Parse JSON string to dict
        ptm = json.loads(ptm_json)
        
        # Map residue index if mapping is provided
        if residue_mapping and 'res_idx' in ptm:
            # Construct residue ID as it would appear in the mapping (assuming chain A)
            res_id = f"A{ptm['res_idx']}"
            
            # Look up the residue ID in the mapping
            if res_id in residue_mapping:
                # Get mapped residue ID
                mapped_res_id = residue_mapping[res_id]
                
                # Extract number (skip the chain letter)
                mapped_num = int(mapped_res_id[1:])
                
                # Update the res_idx with the mapped value
                ptm['res_idx'] = mapped_num
                logger.info(f"Mapped PTM residue from {res_id} to {mapped_res_id}")
            else:
                logger.warning(f"No mapping found for PTM residue {res_id}. Using original.")
        
        return ptm
    except Exception as e:
        logger.error(f"Error parsing PTM: {e}")
        return None

def process(
    row: pd.Series, 
    output_path: str, 
    ligand_data: Optional[Dict[str, Any]] = None,
    bonds: Optional[List[List[Any]]] = None,
    ptm: Optional[Dict[str, Any]] = None
) -> None:
    """
    Process a FASTA file to JSON format with optional ligands, bonds, and PTM.
    
    Args:
        fasta_path: Path to the FASTA file
        output_path: Path where to save the JSON file
        ligand_data: Optional dictionary with ligand information
        bonds: Optional list of bond specifications
        ptm: Optional dictionary with PTM information
    """
    # Read sequence from FASTA file
    sequence = row['seq']
    logger.info(f"Processing {row['name']} with sequence length {len(sequence)}")
    
    # Get base name for JSON
    fasta_name = row['name']
    
    # Generate 5 random seeds
    num_seeds = 5
    random_seeds = [int(np.random.randint(0, 2**32-1)) for _ in range(num_seeds)]    
    # Initialize JSON data structure
    json_data = {
        'name': fasta_name,
        'sequences': [
            {
                'protein': {
                    'id': ['A'],
                    'sequence': sequence,
                    'templates': [],
                    'unpairedMsa': '',
                    'pairedMsa': ''
                }
            }
        ],
        'modelSeeds': random_seeds,
        'dialect': 'alphafold3',
        'version': 2
    }
    
    # Add PTM information to protein if provided
    if ptm and 'cid' in ptm and 'res_idx' in ptm:
        # Add modifications array to the protein object
        protein_data = json_data['sequences'][0]['protein']
        protein_data['modifications'] = [
            {'ptmType': ptm['cid'], 'ptmPosition': ptm['res_idx']}
        ]
    
    # Add ligand data if provided
    # pdb.set_trace()
    if ligand_data:
        ligand_data = ligand_data[0]
        # For a single CIF file
        if 'cif' in ligand_data and ligand_data['cif']:  # Check if exists and not empty
            ligand_path = ligand_data['cif']
            if os.path.exists(ligand_path):
                with open(ligand_path, 'r') as f:
                    ligand_cif_content = f.read()
            else:
                error_msg = f"Ligand CIF file not found: {ligand_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Only add ligand sequence if id exists and is not empty
        if 'id' in ligand_data and ligand_data['id']:  # Check if exists and not empty
            # Add ligand value to sequences
            json_data['sequences'].append({
                'ligand': {
                    'id': ligand_data.get('id', []),
                    'ccdCodes': ligand_data.get('ccdCodes', [])
                }
            })
        
        # Only add userCCD if we actually read some content
        if ligand_cif_content:
            json_data['userCCD'] = ligand_cif_content
    
    # Add bonds if provided
    if bonds:
        json_data["bondedAtomPairs"] = bonds
    
    # pdb.set_trace()
    # Write JSON to file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Successfully processed {row['name']}")

def main():
    """Main function to process FASTA files to JSON format."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_df = pd.read_csv(args.input_csv)
    # Process only the specified range
    end_idx = len(input_df) if args.num_files == -1 else min(args.start_idx + args.num_files, len(input_df))
    seqs_to_process_df = input_df.iloc[args.start_idx:end_idx]
    
    logger.info(f"Processing {len(seqs_to_process_df)} sequences...")
    
    # Parse ligand data and bonds if provided
    ligand_data = json.loads(args.ligand_data) if args.ligand_data else None
    
    # Process each FASTA file
    for index, row in seqs_to_process_df.iterrows():
        design_name = row['name']
        output_path = os.path.join(args.output_dir, f"{design_name}.json")
        
        # Parse bonds and PTM if provided (with CSV mapping)
        bonds = parse_bonds(args.bonds, row) if args.bonds else None
        ptm = parse_ptm(args.ptm) if args.ptm else None
        
        process(
            row=row, 
            output_path=output_path,
            ligand_data=ligand_data,
            bonds=bonds,
            ptm=ptm
        )
    
    logger.info(f"Successfully processed {len(seqs_to_process_df)} files")

if __name__ == "__main__":
    main()