import torch
from sklearn.metrics import r2_score
import numpy as np

import matplotlib.pyplot as plt
import wandb

def get_r2(ab_true:(torch.Tensor), ab_pred:(torch.Tensor), mode:(str)) -> dict:
    """
    Takes in the true and predicted abundances, returns a list of R^2 related stuff and linreg related stuff.
    Both ab_true and ab_pred must be detached

    Returns:
        r2_metrics (dict): The r2 values in keys 'total', 'gv', 'npv', 'soil'
    """
    if mode not in ['val', 'test']:
        raise ValueError(f"Unknown/Unsupported r2 mode: {mode}")
    em_list = [f'general/{mode}_gv_r2', f'general/{mode}_npv_r2', f'general/{mode}_soil_r2']

    ab_true = ab_true.cpu().numpy()
    ab_pred = ab_pred.cpu().numpy()
    temp = r2_score(ab_true, ab_pred, multioutput='raw_values').tolist()

    r2_metrics = {f'general/{mode}_total_r2': np.mean(temp)}
    for i, j in zip(em_list, temp):
        r2_metrics[i] = j

    return r2_metrics

def plot_abundances(ab_true:(torch.Tensor), ab_pred:(torch.Tensor), mode:(str)) -> None:
    """
    Plots the true vs predicted abundances and saves the figure to the specified path.

    Args:
        ab_true (torch.Tensor): The true abundances.
        ab_pred (torch.Tensor): The predicted abundances.
        mode (str): The mode of operation, e.g., 'val' or 'test'.
    """

    ab_true = ab_true.cpu().numpy()
    ab_pred = ab_pred.cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(ab_true[:,0], ab_pred[:,0], alpha=0.5, label='GV', color='green')
    ax.scatter(ab_true[:,1], ab_pred[:,1], alpha=0.5, label='NPV', color='orange')
    ax.scatter(ab_true[:,2], ab_pred[:,2], alpha=0.5, label='Soil', color='brown')
    ax.plot([0, 1], [0, 1], '--', label='Goal Line', color='black')  # goal line
    ax.set_xlabel('True Abundances')
    ax.set_ylabel('Predicted Abundances')
    ax.set_title(f'True vs Predicted Abundances (in {mode})')
    ax.legend()
    ax.grid()

    wandb.log({f"general/{mode}_abundance_plot": wandb.Image(fig)})

    plt.close(fig)