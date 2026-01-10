import torch

"""
For details of the interpretations of delta, epsilon, zeta, and gammas; check the README.md in this directory
"""

def replace_with_1(tensor:(torch.Tensor), method:(str)='min', dim:(int)=0):
    """
    This function replaces the entries of a tensor, along a dim given a tensor. Example is, whne method is min,
    along a row of the tensor, the minimum value is replaced with 1 while all others are 0.

    Args:
        tensor (torch.Tensor): The tensor to apply the operation on
        method (str): The masker, either 'min' or 'max'
        dim (int): Along which dim to do this, usually 0
    """

    if method == 'min':
        _, idx = tensor.min(dim=dim)
    elif method == 'max':
        _, idx = tensor.max(dim=dim)

    out = torch.zeros_like(tensor)
    out.scatter_(dim, idx.unsqueeze(dim), 1.0)

    return out

def get_metrics_distance(ground_data:(torch.Tensor), synth_data:(torch.Tensor), type:(str)='sam', method:(str)='min', spec_idx:(list[int])=None, p:(float)=2) -> float | int | int | int | int | int | int:
    """
    Gets the similarity metric using euclidian distance.

    Args:
        ground_data (torch.Tensor): Ground truth data, lab data
        synth_data (torch.Tensor): Synthesized data, using abundances
        type (str): Type of distance, euclidian ('euc') or spectral angle mapper ('sam')
        method (str): Method to prioritize minimum ('min') or maximum ('max')
        spec_idx (list[int]): The indices of the tensors, if limiting the wavelengths
        p (float): Power of the euclidian distance

    Returns:
        delta (float): (gamma_11 + gamma_22) / (gamma_12 + gamma_21)
        epsilon (int): (gamma_11 - gamma_22)
        zeta (int): (gamma_12 - gamma_21)
        gammas (int): gamma_11, gamma_12, gamma_21, gamma_22
    """
    if spec_idx is not None:
        ground_data = ground_data[:, spec_idx[0]:spec_idx[1]]
        synth_data = synth_data[:, spec_idx[0]:spec_idx[1]]

    g = ground_data.shape[0] # These are how much data we have for ground truth and synthesized
    s = synth_data.shape[0]
    t = g+s # Total data amount

    concat_data = torch.vstack((ground_data, synth_data))

    if type == 'euc':
        distance_matrix = torch.cdist(concat_data, concat_data, p) # Gets the distance matrix, here, distance at loc (i,j) is from point i to j
    elif type == 'sam':
        norms = concat_data.norm(p=2, dim=1, keepdim=True)
        normed = concat_data / (norms + 1e-6)
        distance_matrix = torch.mm(normed, normed.t())
        distance_matrix = torch.acos(torch.clamp(distance_matrix, -1 + 1e-6, 1 - 1e-6)) # Gets the distance matrix, same indexing as euc, range is [0, pi]
    else:
        raise ValueError(f"Unknown/Unsupported distance type: {type}")
    
    if method == 'min':
        distance_matrix.fill_diagonal_(float('inf')) # Fill the diagonal with inf, the reason why we do this is obvious, so that it does not appear as min
    elif method == 'max':
        distance_matrix.fill_diagonal_(float('-inf')) # Fill the diagonal with -inf, the reason why we do this is obvious, so that it does not appear as max
    else:
        raise ValueError(f"Unknown/Unsupported method: {method}")

    distance_matrix = replace_with_1(distance_matrix, method, 0) # Replace along rows, the minimum (or maximum) with 1, others with 0

    gamma_11 = int(torch.sum(torch.sum(distance_matrix[:g, :g])).item())
    gamma_12 = int(torch.sum(torch.sum(distance_matrix[:g, g:])).item())
    gamma_21 = int(torch.sum(torch.sum(distance_matrix[g:, :g])).item())
    gamma_22 = int(torch.sum(torch.sum(distance_matrix[g:, g:])).item())

    gamma_similar = gamma_11 + gamma_22 # Sum along the diagonal quartiles, within these, the datapoints are closest to ground (if itself is ground) and the conjugate
    gamma_different = gamma_12 + gamma_21 # Sum along inverse diag, this means they are mixed among each other

    if gamma_different.item() != 0: # As long as we have some different points, use regular
        delta = (gamma_similar / gamma_different).item()
    else:
        delta = float('inf') # Otherwise return infinity

    epsilon = gamma_11 - gamma_22 # The clustering 'measure'
    zeta = gamma_12 - gamma_21 # Density 'measure'

    return delta, epsilon, zeta, gamma_11, gamma_12, gamma_21, gamma_22