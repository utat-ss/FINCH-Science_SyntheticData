import torch


def get_unnormalizer(data_norm_dict:(dict)):
    """
    The function to get the unnormalizer function of a given data normalizer.
    """

    if data_norm_dict['norm_type']=='classic':
        return lambda normed_data: (normed_data+1)/2
    elif data_norm_dict['norm_type']=='dynamic':
        max_vals, min_vals = data_norm_dict['max_vals'], data_norm_dict['min_vals']
        return lambda normed_data: ((normed_data + 1)*(max_vals - min_vals))/2 + min_vals
    elif data_norm_dict['norm_type']=='log':
        max_vals, min_vals, eps = data_norm_dict['max_vals'], data_norm_dict['min_vals'], data_norm_dict['eps']
        return lambda normed_data: torch.exp(((normed_data + 1)*(max_vals - min_vals))/2 + min_vals) + eps
    elif data_norm_dict['norm_type']=='none':
        return lambda normed_data: normed_data
    else:
        raise ValueError(f"Unknown/Unsupported normalization type {data_norm_dict['norm_type']}.")


