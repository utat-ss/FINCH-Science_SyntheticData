"""

This is the file to define all the training functions. It includes:
    - Diffusion training function
    - 

"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Tuple, List, Optional

def get_vals(data_handle: str = r'data\simpler_data_rwc.csv', spec_range: List[int] = [400, 2490]) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:

    """
    The function to get all the datasets.

    Parameters:
        - data_handle (str): The dataset handle, path str rooting the github repo
        - range (list[int]): A list of spectral limits

    Outputs:
        - spectra_tensor (torch.Tensor): Spectral val tensor
        - abundances_tensor (torch.Tensor): Abundances tensor
        - names (list): A list of all the spectral 
        - indices (list): A list of the actual indices in a csv file
    """

    # Get the dataframe given the handle
    df = pd.read_csv(data_handle)

    # Get the spectral vals given the spectral range
    spectral_cols = [str(w) for w in range(spec_range[0], spec_range[1] + 1, 10)]
    spectral_cols = [c for c in spectral_cols if c in df.columns]
    spectra = df[spectral_cols].values.astype("float32")
    spectra_tensor = torch.from_numpy(spectra)

    # Get the abundances
    abund_cols = ["gv_fraction", "npv_fraction", "soil_fraction"]
    abund_cols = [c for c in abund_cols if c in df.columns]
    abundances = df[abund_cols].values.astype("float32")
    abundances_tensor = torch.from_numpy(abundances)

    # names and indices
    if "Spectra" in df.columns:
        names = df["Spectra"].to_list()
    else:
        names = [f"row_{i}" for i in range(len(df))]
    indices = list(range(len(names)))

    return spectra_tensor, abundances_tensor, names, indices

class HyperSpectralDataset(Dataset):
    """
    Defining this class so that we can keep track of spectra, abundances, names, and indices.

    Parameters:
        - spectra (torch.Tensor): Tensor of all the spectra, as a tensor
        - abundances (torch.Tensor): Abundances of all the spectra, as a tensor
        - names (list): Names of the spectra, as a list
        - indices (list): Indices in order, as a list
    """
    def __init__(self, spectra: torch.Tensor, abundances: torch.Tensor, names: List[str], indices: List[int]):
        self.spectra = spectra
        self.abundances = abundances
        self.names = names
        self.indices = indices

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return {
            'spectrum': self.spectra[idx],
            'abundances': self.abundances[idx],
            'name': self.names[idx],
            'orig_index': self.indices[idx]
        }

def get_dataloaders(ds: HyperSpectralDataset, cfg_loader: dict):
    """
    Gets the dataloaders for train, test, validate datasets.

    Parameters:
        - ds (HyperSpectralDataset): The entire dataset as a HyperSpectralDataset class.
        - cfg_loader (dict):
            - 'test' (int): number of test samples
            - 'validate' (int): number of validation samples
            - 't_batch' (int): train batch size
            - 'seed' (int): optional RNG seed
    """

    # Take in the values from the dict
    n = len(ds)
    n_test = int(cfg_loader.get('test', 23))
    n_validate = int(cfg_loader.get('validate', 4))
    train_batch = int(cfg_loader.get('t_batch', 1))
    seed = cfg_loader.get('seed', 0)

# Infer the amount of n_train, ensure completeness
    assert n_test + n_validate < n, "Not enough samples for requested splits"

    n_train = n - n_test - n_validate
    lengths = [n_train, n_validate, n_test]
    generator = torch.Generator().manual_seed(seed)

    # Separate the temp dataset into train and test
    ds_train, ds_validate, ds_test = random_split(ds, lengths, generator=generator)

    dl_train = DataLoader(ds_train, batch_size=train_batch, shuffle=True)
    dl_val = DataLoader(ds_validate, batch_size=1, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    return dl_train, dl_val, dl_test

def train_diffusion(cfg_train: dict, cond_diffusion, loss, optimizer: torch.optim.Optimizer, data_handle: Optional[str] = None):
    """
    The function to train and validate the models, takes in:
        - cfg_train (dict): 
            - cfg_loader (dict):
                - 'test' (int): How many samples to be used to test at the end
                - 't_batch' (int): How many train batches in total, to be trained on
                - 'validate' (int): How many samples to validate on each epoch
                - 'epoch' (int): How many epochs in total
            - 'range' (list): Spectral range [lower, upper]
            - 'device' : The device being used
        - cond_diffusion (class): The conditional diffusion class, already initialized
        - loss (class): Loss function class, must have MSE and SAM at the very least
        - optimizer (torch.optim): The optimizer to use
        - data_handle (str): The path handle to the dataset
    """

    device = cfg_train.get('device', 'cpu')
    device = torch.device(device)

        ### DATA COLLECTION ###

    # First, we will pre-process and parse all the data
    # Get the spectral range that we want to use
    spec_range = cfg_train.get('range', [400, 2490])
    data_path = data_handle or cfg_train.get('data_handle', r'data\simpler_data_rwc.csv')

    # Parse the csv into sub-parts we want to use
    spectra, abundances, names, indices = get_vals(data_handle=data_path, spec_range=spec_range)

    # Generate the hyperspectral dataset
    ds = HyperSpectralDataset(spectra=spectra, abundances=abundances, names=names, indices=indices)

    # Dataload the giant dataset into train, test, validation datasets
    cfg_loader = cfg_train.get('cfg_loader', {'test': 23, 't_batch': 1, 'validate': 4, 'seed': 0})

    # Get the dataloaded datasets
    dl_train, dl_validate, dl_test = get_dataloaders(ds, cfg_loader)

    epochs = int(cfg_loader.get('epoch', cfg_train.get('epochs', 50)))
    unconditional_prob = cfg_train.get('unconditional_prob', 0.0)

    # training utilities from cfg
    use_amp = cfg_train.get('use_amp', False)
    clip_grad = cfg_train.get('clip_grad', 1.0)
    ema_every = cfg_train.get('ema_every', 1)
    save_every = cfg_train.get('save_every', 1000)
    ckpt_dir = cfg_train.get('ckpt_dir', './checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    ### TRAINING, VALIDATION, TEST LOOP ###

    # Define the init dictionary where we will keep track of the generated spectra during val and test. Also, create these lists to collect all the losses
    collector_dict = {'gen_spec': {'validate': {}, 'test': {}},
                      'losses': {'train': [], 'val': [], 'test': []},
                      'loss_by_t': None}

    # prepare timestep accumulators if scheduler has steps
    T = getattr(cond_diffusion.scheduler, "steps", None)
    if T is not None:
        loss_by_t_sum = torch.zeros(T + 1, dtype=torch.float64)
        loss_by_t_count = torch.zeros(T + 1, dtype=torch.long)
        collector_dict['losses']['by_t_sum'] = loss_by_t_sum
        collector_dict['losses']['by_t_count'] = loss_by_t_count

    cond_diffusion.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    for epoch in range(1, epochs + 1):
        ### TRAINING LOOP ###

        # Set the epsilon in training mode
        cond_diffusion.epsilon.train()
        total_train_loss = 0.0 # Accumulate total train loss per epoch
        n_batches = 0

        for batch in dl_train:
            x0 = batch['spectrum'].to(device)
            abund = batch['abundances'].to(device)

            optimizer.zero_grad()

            # training_procedure expected to sample timesteps internally if scheduler supports it
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = cond_diffusion.training_procedure(x0, abund, unconditional_prob=unconditional_prob)
                # unpack robustly
                if len(out) == 5:
                    x0_hat, xn, xn_hat, logvar, t = out
                elif len(out) == 4:
                    x0_hat, xn, xn_hat, t = out
                    logvar = None
                else:
                    x0_hat, xn, xn_hat = out
                    logvar = None
                    t = None

                # clamp predicted logvar for stability if present
                if logvar is not None:
                    logvar = torch.clamp(logvar, -20.0, 20.0)

                # loss supports hybrid combination internally if configured
                total_loss = loss(x0, x0_hat, xn, xn_hat, logvar) if (logvar is not None) else loss(x0, x0_hat, xn, xn_hat)

            # backward / step with scaler
            scaler.scale(total_loss).backward()
            # unscale for gradient clipping
            scaler.unscale_(optimizer)
            if clip_grad is not None and clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(cond_diffusion.epsilon.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            if (step % max(1, ema_every)) == 0:
                try:
                    cond_diffusion.update_ema()
                except Exception:
                    pass

            # bookkeeping
            loss_val = float(total_loss.detach().cpu().item())
            total_train_loss += loss_val
            collector_dict['losses']['train'].append(loss_val)
            n_batches += 1

            # per-timestep accumulation
            if (t is not None) and (hasattr(cond_diffusion.scheduler, "steps")):
                # t can be tensor shape (B,)
                if isinstance(t, torch.Tensor):
                    t_cpu = t.detach().cpu()
                    if t_cpu.ndim == 0:
                        idxs = [int(t_cpu.item())]
                        vals = [loss_val]
                    else:
                        # distribute loss equally across batch entries (approx)
                        for tt in t_cpu:
                            idx = int(tt.item())
                            collector_dict['losses']['by_t_sum'][idx] += loss_val / float(t_cpu.numel())
                            collector_dict['losses']['by_t_count'][idx] += 1
                else:
                    idx = int(t)
                    collector_dict['losses']['by_t_sum'][idx] += loss_val
                    collector_dict['losses']['by_t_count'][idx] += 1

            step += 1

            # optional checkpoint save
            if (step % save_every) == 0:
                ckpt = {
                    "step": step,
                    "model_state": cond_diffusion.epsilon.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "ema_params": getattr(cond_diffusion, "ema_params", None),
                    "cfg_train": cfg_train
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_{step}.pth"))

        avg_train = total_train_loss / max(1, n_batches)
        print(f"Epoch {epoch} | Average Training Loss: {avg_train:.6f}")

        # Validation using EMA weights (if available)
        cond_diffusion.epsilon.eval()
        total_val_loss = 0.0
        actual_list = []
        abundance_list = []
        generated_list = []

        # if EMA exists, swap in ema params temporarily for sampling
        ema_params = getattr(cond_diffusion, "ema_params", None)
        if ema_params is not None:
            # save current state
            saved_state = cond_diffusion.epsilon.state_dict()
            # load ema
            cond_diffusion.load_ema_to_model()

        with torch.no_grad():
            for batch in dl_validate:
                x0 = batch['spectrum'].to(device)
                abund = batch['abundances'].to(device)

                sample_out = cond_diffusion.sample(ab=abund)
                if isinstance(sample_out, tuple):
                    x0_pred = sample_out[0]
                else:
                    x0_pred = sample_out
                x0_pred = x0_pred.to(device)

                val_loss = loss.recons_loss(x0, x0_pred) if hasattr(loss, "recons_loss") else torch.tensor(0.0, device=device)
                total_val_loss += float(val_loss.detach().cpu().item())
                collector_dict['losses']['val'].append(float(val_loss.detach().cpu().item()))

                actual_list.append(x0.detach().cpu())
                abundance_list.append(abund.detach().cpu())
                generated_list.append(x0_pred.detach().cpu())

        # restore model weights if EMA swapped
        if ema_params is not None:
            cond_diffusion.epsilon.load_state_dict(saved_state)

        if len(dl_validate) > 0:
            avg_val = total_val_loss / len(dl_validate)
        else:
            avg_val = 0.0
        collector_dict['gen_spec']['validate'][epoch] = {
            'actual': torch.cat(actual_list, dim=0) if actual_list else torch.empty(0),
            'abundances': torch.cat(abundance_list, dim=0) if abundance_list else torch.empty(0),
            'generated': torch.cat(generated_list, dim=0) if generated_list else torch.empty(0)
        }
        print(f"Epoch {epoch} | Average Validation Loss: {avg_val:.6f}")

    # Testing (use EMA weights for sampling if present)
    cond_diffusion.epsilon.eval()
    total_test_loss = 0.0
    actual_list = []
    abundance_list = []
    generated_list = []

    ema_params = getattr(cond_diffusion, "ema_params", None)
    if ema_params is not None:
        saved_state = cond_diffusion.epsilon.state_dict()
        cond_diffusion.load_ema_to_model()

    with torch.no_grad():
        for batch in dl_test:
            x0 = batch['spectrum'].to(device)
            abund = batch['abundances'].to(device)

            sample_out = cond_diffusion.sample(ab=abund)
            if isinstance(sample_out, tuple):
                x0_pred = sample_out[0]
            else:
                x0_pred = sample_out
            x0_pred = x0_pred.to(device)

            test_loss = loss.recons_loss(x0, x0_pred) if hasattr(loss, "recons_loss") else torch.tensor(0.0, device=device)
            total_test_loss += float(test_loss.detach().cpu().item())
            collector_dict['losses']['test'].append(float(test_loss.detach().cpu().item()))

            actual_list.append(x0.detach().cpu())
            abundance_list.append(abund.detach().cpu())
            generated_list.append(x0_pred.detach().cpu())

    if ema_params is not None:
        cond_diffusion.epsilon.load_state_dict(saved_state)

    if actual_list:
        collector_dict['gen_spec']['test'] = {
            'actual': torch.cat(actual_list, dim=0),
            'abundances': torch.cat(abundance_list, dim=0),
            'generated': torch.cat(generated_list, dim=0)
        }
    avg_test = total_test_loss / max(1, len(dl_test))
    print(f"Average Test Loss: {avg_test:.6f}")

    # finalize per-timestep stats
    if ('losses' in collector_dict) and ('by_t_sum' in collector_dict['losses']):
        by_t_sum = collector_dict['losses']['by_t_sum']
        by_t_count = collector_dict['losses']['by_t_count']
        # avoid division by zero
        by_t_mean = torch.zeros_like(by_t_sum)
        nonzero = by_t_count > 0
        by_t_mean[nonzero] = (by_t_sum[nonzero] / by_t_count[nonzero].float())
        collector_dict['losses']['by_t_mean'] = by_t_mean

    # Convert losses to tensors
    collector_dict['losses']['train'] = torch.tensor(collector_dict['losses']['train'], dtype=torch.float32) if collector_dict['losses']['train'] else torch.empty(0)
    collector_dict['losses']['val'] = torch.tensor(collector_dict['losses']['val'], dtype=torch.float32) if collector_dict['losses']['val'] else torch.empty(0)
    collector_dict['losses']['test'] = torch.tensor(collector_dict['losses']['test'], dtype=torch.float32) if collector_dict['losses']['test'] else torch.empty(0)

    return collector_dict, dl_train, dl_validate, dl_test, cond_diffusion
