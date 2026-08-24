import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def plot_rotations_as_euler(
    rotations: list[R],
    axis_spec: str,
    axes: plt.Axes,
    rows: list[int],
    cols: list[int],
    degrees=False,
    hist_kwargs=dict(bins=100),
    scatter_kwargs=dict(s=0.5, alpha=0.5),
):
    euler_angles = [rot.as_euler(axis_spec, degrees=degrees) for rot in rotations]
    euler_angles = list(zip(*euler_angles))
    euler_X = euler_angles[0]
    euler_Y = euler_angles[1]
    euler_Z = euler_angles[2]
    euler_range = (-180, 180) if degrees else (-math.pi, math.pi)
    hist_kwargs = {**dict(range=euler_range), **hist_kwargs}
    axes[rows[0], cols[0]].hist(euler_X, **hist_kwargs)
    axes[rows[0], cols[0]].set_title(f"Euler {axis_spec[0]}")
    axes[rows[0], cols[1]].hist(euler_Y, **hist_kwargs)
    axes[rows[0], cols[1]].set_title(f"Euler {axis_spec[1]}")
    axes[rows[0], cols[2]].hist(euler_Z, **hist_kwargs)
    axes[rows[0], cols[2]].set_title(f"Euler {axis_spec[2]}")

    axes[rows[1], cols[0]].scatter(euler_X, euler_Y, **scatter_kwargs)
    axes[rows[1], cols[0]].set_xlabel(f"Euler {axis_spec[0]}")
    axes[rows[1], cols[0]].set_ylabel(f"Euler {axis_spec[1]}")
    axes[rows[1], cols[0]].set_title(f"Euler {axis_spec[0]} vs {axis_spec[1]}")
    axes[rows[1], cols[0]].set_xlim(euler_range)
    axes[rows[1], cols[0]].set_ylim(euler_range)
    axes[rows[1], cols[1]].scatter(euler_Y, euler_Z, **scatter_kwargs)
    axes[rows[1], cols[1]].set_xlabel(f"Euler {axis_spec[1]}")
    axes[rows[1], cols[1]].set_ylabel(f"Euler {axis_spec[2]}")
    axes[rows[1], cols[1]].set_title(f"Euler {axis_spec[1]} vs {axis_spec[2]}")
    axes[rows[1], cols[1]].set_xlim(euler_range)
    axes[rows[1], cols[1]].set_ylim(euler_range)
    axes[rows[1], cols[2]].scatter(euler_Z, euler_X, **scatter_kwargs)
    axes[rows[1], cols[2]].set_xlabel(f"Euler {axis_spec[2]}")
    axes[rows[1], cols[2]].set_ylabel(f"Euler {axis_spec[0]}")
    axes[rows[1], cols[2]].set_title(f"Euler {axis_spec[2]} vs {axis_spec[0]}")
    axes[rows[1], cols[2]].set_xlim(euler_range)
    axes[rows[1], cols[2]].set_ylim(euler_range)


def plot_rotations_as_quat(
    rotations: list[R],
    axes: plt.Axes,
    rows: list[int],
    cols: list[int],
    hist_kwargs=dict(bins=100),
    scatter_kwargs=dict(s=0.5, alpha=0.5),
):
    quats = [rot.as_quat() for rot in rotations]
    quats = list(zip(*quats))
    qx = quats[0]
    qy = quats[1]
    qz = quats[2]
    qw = quats[3]
    axes[rows[0], cols[0]].hist(qx, range=(-1, 1), **hist_kwargs)
    axes[rows[0], cols[0]].set_title("Quaternion x")
    axes[rows[0], cols[0]].set_xlim(-1, 1)
    axes[rows[0], cols[1]].hist(qy, range=(-1, 1), **hist_kwargs)
    axes[rows[0], cols[1]].set_title("Quaternion y")
    axes[rows[0], cols[1]].set_xlim(-1, 1)
    axes[rows[0], cols[2]].hist(qz, range=(-1, 1), **hist_kwargs)
    axes[rows[0], cols[2]].set_title("Quaternion z")
    axes[rows[0], cols[2]].set_xlim(-1, 1)
    axes[rows[0], cols[3]].hist(qw, range=(-1, 1), **hist_kwargs)
    axes[rows[0], cols[3]].set_title("Quaternion w")
    axes[rows[0], cols[3]].set_xlim(0, 1)

    axes[rows[1], cols[0]].scatter(qx, qy, **scatter_kwargs)
    axes[rows[1], cols[0]].set_xlabel("Quaternion x")
    axes[rows[1], cols[0]].set_ylabel("Quaternion y")
    axes[rows[1], cols[0]].set_title("Quaternion x vs y")
    axes[rows[1], cols[0]].set_xlim(-1, 1)
    axes[rows[1], cols[0]].set_ylim(-1, 1)
    axes[rows[1], cols[1]].scatter(qy, qz, **scatter_kwargs)
    axes[rows[1], cols[1]].set_xlabel("Quaternion y")
    axes[rows[1], cols[1]].set_ylabel("Quaternion z")
    axes[rows[1], cols[1]].set_title("Quaternion y vs z")
    axes[rows[1], cols[1]].set_xlim(-1, 1)
    axes[rows[1], cols[1]].set_ylim(-1, 1)
    axes[rows[1], cols[2]].scatter(qx, qz, **scatter_kwargs)
    axes[rows[1], cols[2]].set_xlabel("Quaternion x")
    axes[rows[1], cols[2]].set_ylabel("Quaternion z")
    axes[rows[1], cols[2]].set_title("Quaternion x vs z")
    axes[rows[1], cols[2]].set_xlim(-1, 1)
    axes[rows[1], cols[2]].set_ylim(-1, 1)
    axes[rows[2], cols[0]].scatter(qx, qw, **scatter_kwargs)
    axes[rows[2], cols[0]].set_xlabel("Quaternion x")
    axes[rows[2], cols[0]].set_ylabel("Quaternion w")
    axes[rows[2], cols[0]].set_title("Quaternion x vs w")
    axes[rows[2], cols[0]].set_xlim(-1, 1)
    axes[rows[2], cols[0]].set_ylim(0, 1)
    axes[rows[2], cols[1]].scatter(qy, qw, **scatter_kwargs)
    axes[rows[2], cols[1]].set_xlabel("Quaternion y")
    axes[rows[2], cols[1]].set_ylabel("Quaternion w")
    axes[rows[2], cols[1]].set_title("Quaternion y vs w")
    axes[rows[2], cols[1]].set_xlim(-1, 1)
    axes[rows[2], cols[1]].set_ylim(0, 1)
    axes[rows[2], cols[2]].scatter(qz, qw, **scatter_kwargs)
    axes[rows[2], cols[2]].set_xlabel("Quaternion z")
    axes[rows[2], cols[2]].set_ylabel("Quaternion w")
    axes[rows[2], cols[2]].set_title("Quaternion z vs w")
    axes[rows[2], cols[2]].set_xlim(-1, 1)
    axes[rows[2], cols[2]].set_ylim(0, 1)
