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