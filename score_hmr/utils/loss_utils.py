
import torch
import numpy as np


'''
def keyp_rep_error_l1(smpl_keyp_2d, keyp_hourglass, keyp_hourglass_scores, thr_kp=0.3):
    # step 1: make sure that the hg prediction and barc are close
    with torch.no_grad():
        kp_weights = keyp_hourglass_scores
        kp_weights[keyp_hourglass_scores<thr_kp] = 0
    loss_keyp_rep = torch.mean((torch.abs((smpl_keyp_2d - keyp_hourglass)/512)).sum(dim=2)*kp_weights[:, :, 0])
    return loss_keyp_rep

def keyp_rep_error(smpl_keyp_2d, keyp_hourglass, keyp_hourglass_scores, thr_kp=0.3):
    # step 1: make sure that the hg prediction and barc are close
    with torch.no_grad():
        kp_weights = keyp_hourglass_scores
        kp_weights[keyp_hourglass_scores<thr_kp] = 0
    # losses['kp_reproj']['value'] = torch.mean((((smpl_keyp_2d - keyp_reproj_init)/512)**2).sum(dim=2)*kp_weights[:, :, 0])
    loss_keyp_rep = torch.mean((((smpl_keyp_2d - keyp_hourglass)/512)**2).sum(dim=2)*kp_weights[:, :, 0])
    return loss_keyp_rep
'''


def limb_sideway_error(optimed_pose_with_glob):
    assert optimed_pose_with_glob.shape[1] == 24

    arm_indices_right = np.asarray([17, 19, 21, 23])
    arm_indices_left = np.asarray([16, 18, 20, 22])
    leg_indices_right = np.asarray([2, 5, 8, 11])
    leg_indices_left = np.asarray([1, 4, 7, 10])

    x0_rotmat = optimed_pose_with_glob   # (1, 24, 3, 3)
    x0_rotmat_arms_left = x0_rotmat[:, arm_indices_left, :, :]
    x0_rotmat_arms_right = x0_rotmat[:, arm_indices_right, :, :]
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]

    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec[2] = -1
    x0_arms_left = x0_rotmat_arms_left.reshape((-1, 3, 3)) @ vec
    x0_arms_right = x0_rotmat_arms_right.reshape((-1, 3, 3)) @ vec
    x0_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3)) @ vec
    x0_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3)) @ vec
    loss_pose_legs_side = (x0_arms_left[:, 1] ** 2).mean() + (x0_arms_right[:, 1] ** 2).mean() + \
                          (x0_legs_left[:, 1] ** 2).mean() + (x0_legs_right[:, 1] ** 2).mean()
    return loss_pose_legs_side


def limb_torsion_error(optimed_pose_with_glob):
    assert optimed_pose_with_glob.shape[1] == 24

    arm_indices_right = np.asarray([17, 19, 21, 23])
    arm_indices_left = np.asarray([16, 18, 20, 22])
    leg_indices_right = np.asarray([2, 5, 8, 11])
    leg_indices_left = np.asarray([1, 4, 7, 10])

    x0_rotmat = optimed_pose_with_glob   # (1, 24, 3, 3)
    x0_rotmat_arms_left = x0_rotmat[:, arm_indices_left, :, :]
    x0_rotmat_arms_right = x0_rotmat[:, arm_indices_right, :, :]
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]

    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec_x[0] = 1      # in x direction
    x_x_arms_left = x0_rotmat_arms_left.reshape((-1, 3, 3)) @ vec_x
    x_x_arms_right = x0_rotmat_arms_right.reshape((-1, 3, 3)) @ vec_x
    x_x_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3)) @ vec_x
    x_x_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3)) @ vec_x

    loss_pose_legs_torsion = (x_x_arms_left[:, 1] ** 2).mean() + (x_x_arms_right[:, 1] ** 2).mean() + \
                             (x_x_legs_left[:, 1] ** 2).mean() + (x_x_legs_right[:, 1] ** 2).mean()
    return loss_pose_legs_torsion


def compute_edge_loss(verts, faces):
    """
    verts: (N, V, 3) tensor of vertex positions
    faces: (F, 3) tensor of face indices
    """
    # Gather vertices for each face's edges
    faces = faces.to(torch.long)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    edge1 = verts[:, v0] - verts[:, v1]  # edge between v0 and v1
    edge2 = verts[:, v1] - verts[:, v2]  # edge between v1 and v2
    edge3 = verts[:, v2] - verts[:, v0]  # edge between v2 and v0

    # Compute edge lengths
    edge_lengths = torch.cat([
        torch.norm(edge1, dim=-1, keepdim=True),
        torch.norm(edge2, dim=-1, keepdim=True),
        torch.norm(edge3, dim=-1, keepdim=True)
    ], dim=-1)

    # Edge loss is the variance of edge lengths
    edge_loss = edge_lengths.var()
    return edge_loss


def compute_normal_consistency(verts, faces):
    """
    verts: (N, V, 3) tensor of vertex positions
    faces: (F, 3) tensor of face indices
    """
    # Compute face normals
    faces = faces.to(torch.long)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    face_normals = torch.cross(verts[:, v1] - verts[:, v0], verts[:, v2] - verts[:, v0], dim=-1)
    face_normals = face_normals / (torch.norm(face_normals, dim=-1, keepdim=True) + 1e-8)

    # Find adjacent faces by shared edges
    F = faces.size(0)
    adj_matrix = torch.zeros((F, F), dtype=torch.float32, device=verts.device)
    for i in range(3):
        for j in range(i + 1, 3):
            shared_edge = (faces[:, i].unsqueeze(1) == faces[:, j].unsqueeze(0)).float()
            adj_matrix += shared_edge

    # Dot product between adjacent face normals
    normal_consistency = (adj_matrix.unsqueeze(-1) * (face_normals @ face_normals.transpose(1, 2))).mean()
    return 1 - normal_consistency  # Maximize consistency


def compute_laplacian_smoothing(verts, faces):
    """
    verts: (N, V, 3) tensor of vertex positions
    faces: (F, 3) tensor of face indices
    """
    faces = faces.to(torch.long)
    # Create adjacency matrix
    adjacency_matrix = torch.zeros((verts.size(1), verts.size(1)), dtype=torch.float32, device=verts.device)
    for i in range(3):  # Loop over edges
        v0, v1 = faces[:, i], faces[:, (i + 1) % 3]
        adjacency_matrix[v0, v1] = 1
        adjacency_matrix[v1, v0] = 1

    # Normalize adjacency matrix
    degree = adjacency_matrix.sum(dim=-1, keepdim=True)
    laplacian_matrix = adjacency_matrix / (degree + 1e-6)

    # Compute Laplacian smoothing
    smoothed_verts = laplacian_matrix @ verts[0]  # For each vertex, compute weighted average of neighbors
    laplacian_loss = torch.norm(smoothed_verts - verts[0], dim=-1).mean()
    return laplacian_loss




