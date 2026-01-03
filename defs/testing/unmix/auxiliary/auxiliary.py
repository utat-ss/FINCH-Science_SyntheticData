def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def setup_wandb(cfg_run, cfg_export):
    import wandb

    settings = wandb.Settings(
        show_errors=True,
        silent=False,
        show_warnings=True
    )

    run = wandb.init(
        project= cfg_export['project_name'],
        name= cfg_export.get('run_name', 'default'),
        config= cfg_run,
        job_type= 'training',
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
