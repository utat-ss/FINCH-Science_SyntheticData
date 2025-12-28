"""
This file is used to define all the plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_change(spectral_matrix, wavelengths= None, interval= 50, repeat= True, save_path= None):

    n_frames, n_wavelengths = spectral_matrix.shape

    if wavelengths is None:
        wavelengths = np.arange(n_wavelengths)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    ax.set_xlim(wavelengths.min(), wavelengths.max())
    ax.set_ylim(-3,
                3)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance + Noise")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        y = spectral_matrix[frame]
        line.set_data(wavelengths, y)
        ax.set_title(f"Spectrum Evolution (Time {frame+1}/{n_frames})")
        return (line,)

    animate = animation.FuncAnimation(fig, update, frames= n_frames, init_func= init, blit= True, interval= interval, repeat= repeat)

    if save_path:
        animate.save(save_path, writer="pillow" if save_path.endswith(".gif") else "ffmpeg")

    plt.show()

    return animate

import wandb
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_to_wandb(x_real:(torch.Tensor), x_gen:(torch.Tensor), abundances, name, orig_idx, unnorm_func, n_samples:(int), epoch:(int)=None):
    """
    This function plots reconstructed spectra after unnormalizing graph to plotly, which is interactive.
    """

    x_real = x_real.detach().cpu(); x_gen = x_gen.detach().cpu(); abundances = abundances.detach().cpu()
    n_samples = min(n_samples, x_real.shape[0])

    plot_titles = []
    for i in range(n_samples):
        title_parts = [f"Sample {i+1}"]
        
        # Add Abundances, Spectral Name, and Original Index
        fracs = abundances[i]
        ab_str = f"GV:{fracs[0]:.2f} NPV:{fracs[1]:.2f} Soil:{fracs[2]:.2f}"
        title_parts.append(ab_str)
        title_parts.append(f"Name: {name[i]}")
        title_parts.append(f"Idx: {orig_idx[i]}")  
        plot_titles.append(" | ".join(title_parts))

    fig = make_subplots(
        rows=n_samples, cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=[plot_titles[i] for i in range(n_samples)]
    )

    for i in range(n_samples):
        real_plot = unnorm_func(x_real[i]).numpy()
        gen_plot = unnorm_func(x_gen[i]).numpy()
        x_axis = np.arange(len(real_plot))

        # Ground Truth
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=real_plot, mode='lines', name=f'Ground Truth', 
                line=dict(color='black', width=2),
                legendgroup='group1', showlegend=(i==0)
            ), row=i+1, col=1
        )
        # Prediction
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=gen_plot, mode='lines', name=f'Reconstruction', 
                line=dict(color='red', width=2),
                legendgroup='group2', showlegend=(i==0)
            ), row=i+1, col=1
        )

    h = 2000

    if epoch is None:      
        fig.update_layout(
            height=h * n_samples, 
            title_text=f"Test Reconstruction",
            hovermode="x unified",
            template="plotly_white",
            margin=dict(t=50, b=10, l=50, r=50)
        )
        wandb.log({"test/interactive_plot": wandb.Html(fig.to_html())})
    
    else:
        fig.update_layout(
            height=h * n_samples, 
            title_text=f"Epoch {epoch} Reconstruction",
            hovermode="x unified",
            template="plotly_white",
            margin=dict(t=50, b=50, l=50, r=50)
        )

        wandb.log({"val/interactive_plot": wandb.Html(fig.to_html()), "epoch": epoch})

