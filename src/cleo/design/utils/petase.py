"""
PETase-specific structural metrics for the reward pipeline.

Measures catalytic triad geometry (distances, angles, dihedrals between
Ser/His/Asp and substrate ester atoms), oxyanion hole distances, and
ligand RMSD relative to a reference structure. These metrics are
designed for evaluating PETase variants but can serve as a template for
other enzyme active-site metrics.
"""

import numpy as np
import pandas as pd
import torch
import biotite.structure.io as strucio

from cleo.design.utils.geom import torch_get_rmsd, angle_between_three_points, compute_dihedral


def compute_petase_metrics(cif_path, cfg, ref):

    cr_ser = cfg.cat_res_ser
    cr_his = cfg.cat_res_his
    cr_asp = cfg.cat_res_asp
    oxh1 = cfg.oxhres1
    oxh2 = cfg.oxhres2
    ester_atom_list = cfg.ester_atom_list
    dihedrals = cfg.dihedrals

    # ligand RMSD
    # get info for reference str
    # ref = strucio.load_structure(cfg.ref_str) ### now loading in the main function so we don't have to reload it every time
    ref_ca_xyz = torch.tensor(ref[(ref.chain_id == cfg.ref_protein_chain_id) & (ref.atom_name == "CA")].coord)
    ref_ligand = ref[ref.chain_id == cfg.ref_ligand_chain_id]

    ref_ligand_xyz_list = []
    for atom_name in cfg.ref_ligand_atom_names:
        ref_ligand_xyz_list.append(ref_ligand[ref_ligand.atom_name == atom_name].coord[0])
    ref_ligand_xyz = torch.tensor(ref_ligand_xyz_list)


    # load cif
    atom_array = strucio.load_structure(cif_path)

    # get cat res serine atoms
    cat_res_ser_atom_array = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == cr_ser)]
    assert all(cat_res_ser_atom_array.res_name == "SER"), f"Expected SER at residue {cr_ser} in chain A"

    # get cat res histidine atoms
    cat_res_his_atom_array = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == cr_his)]
    assert all(cat_res_his_atom_array.res_name == "HIS"), f"Expected HIS at residue {cr_his} in chain A"

    # get cat res aspartate atoms
    cat_res_asp_atom_array = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == cr_asp)]
    assert all(cat_res_asp_atom_array.res_name == "ASP"), f"Expected ASP at residue {cr_asp} in chain A"

    oxh1_atom_array = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == oxh1)]
    oxh2_atom_array = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == oxh2)]

    ligand_atoms = atom_array[atom_array.chain_id == "B"]

    # compute dihedral angles to determine substrate orientation and choose the orientation with the closest to positive dihedral angle
    dihedral_angles = []
    for d in dihedrals:
        p0 = atom_array[(atom_array.chain_id == d[0][0]) & (atom_array.res_id == d[0][1]) & (atom_array.atom_name == d[0][2])].coord[0]
        p1 = atom_array[(atom_array.chain_id == d[1][0]) & (atom_array.res_id == d[1][1]) & (atom_array.atom_name == d[1][2])].coord[0]
        p2 = atom_array[(atom_array.chain_id == d[2][0]) & (atom_array.res_id == d[2][1]) & (atom_array.atom_name == d[2][2])].coord[0]
        p3 = atom_array[(atom_array.chain_id == d[3][0]) & (atom_array.res_id == d[3][1]) & (atom_array.atom_name == d[3][2])].coord[0]
        angle = compute_dihedral(p0, p1, p2, p3)
        dihedral_angles.append(angle)

    ester_atoms = ester_atom_list[dihedral_angles.index(max(dihedral_angles))]

    oxh1_N_xyz = oxh1_atom_array[oxh1_atom_array.atom_name=="N"].coord[0]
    oxh2_N_xyz = oxh2_atom_array[oxh2_atom_array.atom_name=="N"].coord[0]
    acylox_xyz = ligand_atoms[ligand_atoms.atom_name==ester_atoms[1]].coord[0]

    # compute distances
    acylox_oxh1bbN = np.linalg.norm(acylox_xyz - oxh1_N_xyz)
    acylox_oxh2bbN = np.linalg.norm(acylox_xyz - oxh2_N_xyz)

    # now measure hisNE2_esterox, hisNE2_serOG and his_ser_angle
    esterox_xyz = ligand_atoms[ligand_atoms.atom_name == ester_atoms[2]].coord[0]
    his_NE2_xyz = cat_res_his_atom_array[cat_res_his_atom_array.atom_name == "NE2"].coord[0]
    his_CE1_xyz = cat_res_his_atom_array[cat_res_his_atom_array.atom_name == "CE1"].coord[0]
    ser_OG_xyz = cat_res_ser_atom_array[cat_res_ser_atom_array.atom_name == "OG"].coord[0]

    hisNE2_esterox = np.linalg.norm(his_NE2_xyz - esterox_xyz)
    hisNE2_serOG = np.linalg.norm(his_NE2_xyz - ser_OG_xyz)
    his_ser_angle = angle_between_three_points(his_CE1_xyz, his_NE2_xyz, ser_OG_xyz)

    # ligand rmsd
    # look at the folder, get all cif files
    des = strucio.load_structure(cif_path)
    des_ligand = des[des.chain_id == cfg.des_ligand_chain_id]
    des_ca_xyz = torch.tensor(des[(des.chain_id == cfg.des_protein_chain_id) & (des.atom_name == "CA")].coord)

    rmsd_ca, U = torch_get_rmsd(ref_ca_xyz, des_ca_xyz)

    ligand_rmsds = []
    for order in cfg.design_ligand_atom_name_orders:
        des_ligand_xyz_list = []
        for atom_name in order:
            des_ligand_xyz_list.append(des_ligand[des_ligand.atom_name == atom_name].coord[0])
        des_ligand_xyz = torch.tensor(des_ligand_xyz_list)
        des_ligand_xyz_centered = des_ligand_xyz - des_ca_xyz.mean(dim=0)
        des_ligand_xyz_aligned = torch.einsum('kj,ji->ki', des_ligand_xyz_centered, U) + ref_ca_xyz.mean(dim=0)

        ligand_rmsds.append(torch.norm(des_ligand_xyz_aligned - ref_ligand_xyz, dim=1).mean().item())

    ligand_rmsd = min(ligand_rmsds)

    out = {
        "acylox_oxh1bbN": acylox_oxh1bbN,
        "acylox_oxh2bbN": acylox_oxh2bbN,
        "hisNE2_esterox": hisNE2_esterox,
        "hisNE2_serOG": hisNE2_serOG,
        "his_ser_angle": his_ser_angle,
        "ligand_rmsd": ligand_rmsd
    }
    return out


def add_petase_metrics_to_df(df_input, cfg, step_name="petase_metrics"):
    '''
    Computes PETase metrics for all cif files in the specified af3_cif path
    and adds them to the input dataframe.
    '''

    ref = strucio.load_structure(cfg.ref_str)

    metrics_list = []
    for idx, row in df_input.iterrows():
        cif_path = row[f"{cfg.ref_step}_path"]
        metrics = compute_petase_metrics(cif_path, cfg, ref)

        _tmp = {
            f"{step_name}_{k}": v for k,v in metrics.items()
        }
        _tmp["name"] = row["name"]
        metrics_list.append(_tmp)

    output_df =  pd.DataFrame(metrics_list)
    df_merged = pd.merge(df_input, output_df, on="name", how="inner")

    return df_merged