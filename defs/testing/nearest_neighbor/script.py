import logging
import os
import torch
import json

from ...diffusion.data.data_preperation import get_unnormalizer
from .data.combined import get_combined_data
from .near_neigh.serialized import serialize_nearneigh
from .auxiliary.export import export_metrics


def parse_cfg_dict(cfg_run:(dict)):
    """
    Parser and reformer of cfg_run, passed in from the master script

    Args:
        cfg_run (dict): The dict for nearest neighborhood run

    Returns:
        cfg_import (dict): Includes paths and specifics of importing psi2 and separating it
        cfg_synthesis (dict): Specifics related to synthesizing the data
        cfg_omega (dict): The dict of all omega combinations, to be used as hyperparameters for nearest neighborhood calculations
        cfg_export (dict): The specifics of where to export all the produced metrics
        cfg_blobs (dict): If exists, activates the plotting of blobs, if None, no blobs are made
    """

    cfg_import = cfg_run['cfg_import']
    cfg_synthesis = cfg_run['cfg_synthesis']
    cfg_omega = cfg_run['cfg_omega']
    cfg_export = cfg_run['cfg_export']
    cfg_blobs = cfg_run.get('cfg_blobs', None)

    device_str = cfg_synthesis['device']
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA but is not available.")
    elif device_str != 'cpu':
        raise ValueError(f"Unknown/Unsupported device: {device_str}. Choose one of: 'cuda', 'cpu'")
    cfg_synthesis['device'] = torch.device(device_str)

    with open(cfg_synthesis['norm_dict_path']) as f:
        norm_dict = json.load(f)
    cfg_synthesis['unnormalizer'] = get_unnormalizer(norm_dict)
    cfg_synthesis.pop('norm_dict_path', None)

    with open(cfg_synthesis['cfg_model_setup_path']) as f:
        cfg_model_setup = json.load(f)
    cfg_model_setup['__cfg_dir__'] = os.path.dirname(os.path.abspath(cfg_synthesis['cfg_model_setup_path']))
    cfg_synthesis['model_config_dict'] = cfg_model_setup
    cfg_synthesis.pop('cfg_model_setup_path', None)

    return cfg_import, cfg_synthesis, cfg_omega, cfg_export, cfg_blobs
        
def nearneigh_script(cfg_export_master, cfg_nearneigh):

    """
    Is the main 'script' used to get the nearest neighborhood metrics, this is a functionalized and regularly updated version of code in this file.

    Args:
        cfg_export_master (dict): Export configs obtained from the master cfg, this is different than the one in cfg_unmix
        cfg_nearneigh (dict): Cfgs to run this specific metric, for details, check example_cfg.yaml in testing/master/

    Returns:
        the nearest neighborhood metrics
    """

    # Parse the cfg_nearneigh dict
    cfg_import, cfg_synthesis, cfg_omega, cfg_export, cfg_blobs = parse_cfg_dict(cfg_nearneigh)
    logging.info("Cfgs taken in, parsed")

    # Get the combined data
    data_real, data_synh, separation_len = get_combined_data(cfg_import, cfg_synthesis)
    logging.info("Combined data acquired")

    # Get the nearest neighborhood results, given the omega and data
    metric_dict = serialize_nearneigh(data_real, data_synh, cfg_omega)
    logging.info("Calculated the nearest neighborhood metrics")

    # Export the metrics into a csv
    export_metrics(metric_dict, cfg_export)
    logging.info(f"Metrics have been exported to: {cfg_export['metrics_save']}")
