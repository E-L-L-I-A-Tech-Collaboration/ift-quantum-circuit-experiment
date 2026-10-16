from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.manifold.geometry import Face


def plot_mesh(points: np.ndarray, faces: Sequence[Face], values: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    triangles = np.asarray(faces)
    mesh = ax.tripcolor(points[:, 0], points[:, 1], triangles, values, shading="gouraud")
    ax.triplot(points[:, 0], points[:, 1], triangles, linewidth=0.45)
    fig.colorbar(mesh, ax=ax, label="Local x-state")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("manifold coordinate 1")
    ax.set_ylabel("manifold coordinate 2")
    fig.tight_layout()
    return fig


def plot_series(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=180)
    return fig