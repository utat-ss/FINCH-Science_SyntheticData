"""
This script is the master script, it calls upon nearneigh_script.py and 
unmix_script.py to get the metrics of the dataset and synthesizer

Workflow:

    - Takes in some args, where the yaml looks like:

        cfg_out:
            ...
            the exact information on where to output all of the stuff generated, specifically the paths and wandb info

        cfg_unmix:
            ... see the exact cfg type from cfg_run_example.yaml

            this primarily requires:
                - specifics of the critic model
                - ksi_train (the dataset generated in a previous synthesizing run)
            
            note: no option to call synthesizer_script.py
        
        cfg_nearneigh:
            ... see the exact cfg type from config_run_example.yaml

            this primarily requires:
                - omega combinations, tuple(type, method, optional(p, if type='euc'))
                - blobs, details of the plots to be output, if not, no plots
                - specifics of the model, to generate the data_synth, models will be loaded by using code form synthesis folder code

        parses these into two different configs, where one is for nearneigh_script.py and the other for unmix_script.py

    - calls the nearneigh_script.py using cfg_nearneigh, outputs the results using info from cfg_out

    - calls the unmix_script.py using cfg_unmix, outputs the results using info from cfg_out
"""


import argparse
import os
import yaml

def cfg_from_args():
    """
    Function to get args from the .yaml args given in the script run code
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config.yaml file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    
    with open(args.config, 'r') as f:
        cfg_test = yaml.safe_load(f)

    return cfg_test

def parse_cfg_dict(cfg_test:(dict)):
    """
    Parses the cfg_test obtained from args running the script.

    Args:
        cfg_test (dict): The cfg for the testing script passed in with --config ...

    Returns:
        cfg_export (dict): The specifics of setting up WandB and Logging
        cfg_nearneigh (dict): Details related to nearest neighbor calculations
        cfg_unmix (dict): Details related to training of the critic model and test of the ksi_train that way
    """

    cfg_export = cfg_test['cfg_export']
    cfg_nearneigh = cfg_test.get('cfg_nearneigh', None)
    cfg_unmix = cfg_test.get('cfg_unmix', None)

    return cfg_export, cfg_nearneigh, cfg_unmix

def setup_wandb(cfg_test, cfg_export):
    import wandb

    settings = wandb.Settings(
        show_errors=True,
        silent=False,
        show_warnings=True
    )

    run = wandb.init(
        entity= cfg_export.get('entity', None),
        project= cfg_export['project'],
        name= cfg_export.get('name', 'default'),
        config= cfg_test,
        job_type= 'testing',
        settings=settings
    )

    return run

def setup_logging(cfg_export):
    import logging
    import sys

    local_log = cfg_export['local_log']

    logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(local_log, mode='w'), # 'w' overwrites, 'a' appends
        logging.StreamHandler(sys.stdout)                  # Keeps console output active for W&B
        ]
    )

import logging
import traceback

if __name__ == "__main__":

    # Get the configs
    cfg_test = cfg_from_args()

    cfg_export, cfg_nearneigh, cfg_unmix = parse_cfg_dict(cfg_test)

    run = setup_wandb(cfg_test, cfg_export)
    setup_logging(cfg_export)

    try:
        if (cfg_nearneigh is None) and (cfg_unmix is None):
            raise ValueError("Neighter Nearest-Neigh nor Unmix is enabled as a test mode, enable at least one")

        if cfg_nearneigh is not None:
            from ..nearest_neighbor.script import *
            logging.info("Nearest metric neighborhood calculations are enabled, proceeding")
            nearneigh_script(cfg_export, cfg_nearneigh)
            logging.info("Nearest metric neighborhood calculations succesfully completed")
        else:
            logging.info("CAUTION: Nearest neighborhood metric is disabled")

        if cfg_unmix is not None:
            from ..unmix.script import *
            logging.info("Unmix metric calculations are enabled, proceeding")
            unmix_script(cfg_export, cfg_unmix)
            logging.info("Unmix metric calculations succesfully completed")
        else:
            logging.info("CAUTION: Unmix metric is disabled")

    except Exception as e:
        run.alert(title= "Testing crashed", text=str(e))
        traceback.print_exc()
        raise e
    
    finally:
        run.finish()

   
