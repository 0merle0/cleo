"""
Geometric utilities: RMSD alignment, angle and dihedral calculations.
"""

import numpy as np
import torch


def torch_get_rmsd(a, b, eps=1e-6):
    """Align coordinates *b* onto *a* via SVD and return (RMSD, rotation matrix U).

    Both inputs should be tensors of shape ``(L, 3)``.
    """
    assert a.shape == b.shape, 'make sure tensors are the same size'
    L = a.shape[0]
    assert a.shape == torch.Size([L,3]), 'make sure tensors are in format [L,3]'

    # center to CA centroid
    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)

    # Computation of the covariance matrix
    C = torch.einsum('kj,ji->ki', torch.transpose(b.type(torch.float32),0,1), a.type(torch.float32))

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.linalg.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([3,3])
    d[:,-1] = torch.sign(torch.linalg.det(V)*torch.linalg.det(W))

    # Rotation matrix U
    U = torch.einsum('kj,ji->ki',(d*V),W)

    # Rotate xyz_hal
    rP = torch.einsum('kj,ji->ki',b.type(torch.float32),U.type(torch.float32))

    L = rP.shape[0]
    rmsd = torch.sqrt(torch.sum((rP-a)*(rP-a), axis=(0,1)) / L + eps)

    return rmsd, U

def angle_between_three_points(A, B, C):
    u = A - B
    v = C - B
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical issues
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def compute_dihedral(p0, p1, p2, p3):
    """
    JG: with help from gemini, validated
    Calculate the signed dihedral angle between four points.
    
    Parameters:
    p0, p1, p2, p3: np.ndarray with shape (3,)
    
    Returns:
    angle (float): Dihedral angle in degrees, in range (-180, +180]
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 for projection
    b1 /= np.linalg.norm(b1)

    # Compute perpendicular vectors to the planes
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)

    # Normalize normals
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    # Compute angle using arctangent of sin and cos components
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b1)

    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg