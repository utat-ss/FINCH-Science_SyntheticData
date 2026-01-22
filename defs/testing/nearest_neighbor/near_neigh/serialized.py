"""
The code here serializes the nearest neighbor metric calculations for easier use in the testing scripts.
"""

from .nearest_neighbor import get_metrics_distance
import logging

def serialize_nearneigh(data_real:(list), data_synth:(list), cfg_omega:(dict)) -> dict:
    """
    Serializes the nearest neighborhood metric calculations, given the data and omega

    Args:
        data_real (list): List of real spectra values
        data_synth (list): List of synthesized spectra values
        cfg_omega (dict): Dictionary of all the omega dicts, they are args for getting the metric

    Returns:
        metric_dict (dict): A dictionary of metrics, where keys are omega_i, and vals are the metrics themselves
    """

    # Create the metrics dict to store results
    metric_dict = {}
    # Infer data
    ground_data = data_real[0]
    synth_data = data_synth[0]

    metric_types = ['delta', 'epsilon', 'zeta', 'gamma_11', 'gamma_12', 'gamma_21', 'gamma_22']

    # Loop through all the omega combinations, append them to the metrics dict
    for omega_key in list(cfg_omega.keys()):

        payload = {} # Creates an empty payload dict
        omega = cfg_omega[omega_key] # Infer the omega values
        payload.update(omega) # Add the omega specifics to the payload, used for exporting type, method, etc.

        metrics_temp = list(get_metrics_distance(ground_data, synth_data, **omega)) # Get the metrics, same order as metric_types
        payload.update(dict(zip(metric_types, metrics_temp))) # Add them to the payload dict

        metric_dict[omega_key] = payload # Add the payload to the metric dict

        logging.info(f'{omega_key} metrics: {metrics_temp}')

    return metric_dict



