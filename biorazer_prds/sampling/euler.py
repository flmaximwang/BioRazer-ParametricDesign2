import numpy as np
from scipy.spatial.transform import Rotation as R


_AXIS_TO_VECTOR = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


def sampling_sphere_fibonacci(point_num: int) -> np.ndarray:
    """
    Sample approximately uniform points on the unit sphere using a Fibonacci lattice.

    Parameters
    ----------
    point_num : int
        Number of points to sample.

    Returns
    -------
    np.ndarray
        An array of shape ``(point_num, 3)`` containing xyz coordinates on the
        unit sphere.
    """
    if point_num <= 0:
        raise ValueError(f"point_num must be positive, got {point_num}")

    indices = np.arange(point_num, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (indices + 0.5) / point_num
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = golden_angle * indices

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))


def plot_sampling_points(
    points: np.ndarray,
    fig=None,
    marker_size: float = 0.5,
    marker_opacity: float = 1.0,
    marker_color=None,
    color_scale: str = "Viridis",
    width: int | None = 700,
    height: int | None = 700,
    zoom: float = 1,
    sphere_opacity: float = 0.15,
    show_sphere: bool = True,
    label: str = "Sampling points",
    title: str | None = None,
):
    """
    Visualize sphere sampling points with Plotly.

    Parameters
    ----------
    points : np.ndarray
        Points with shape ``(N, 3)`` or ``(3,)``. They are plotted directly.
    fig : plotly.graph_objects.Figure | None
        Existing Plotly figure to append the points to. If ``None``, a new
        figure is created.
    marker_size : float
        Marker size for sampled points.
    marker_opacity : float
        Marker opacity for sampled points.
    marker_color : Any
        Marker color specification passed to Plotly. It can be a single color
        string such as ``"red"`` or ``"#1f77b4"``, or an array-like object for
        per-point coloring. If ``None``, point indices are used.
    color_scale : str
        Plotly colorscale name used when ``marker_color`` is numeric or
        ``None``.
    width : int | None
        Figure width in pixels. Use ``None`` to let Plotly decide.
    height : int | None
        Figure height in pixels. Use ``None`` to let Plotly decide.
    zoom : float
        Camera distance scaling factor. Larger values zoom out.
    sphere_opacity : float
        Opacity of the reference unit sphere surface.
    show_sphere : bool
        If True, draw a reference unit sphere. When ``fig`` already contains
        traces, the sphere is only added for a newly created figure.
    label : str
        Legend name for the point trace.
    title : str | None
        Figure title. If ``None``, a default title is used.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.

    """
    import plotly.graph_objects as go

    points = _normalize_points(points)
    point_num = len(points)

    sphere_u = np.linspace(0.0, 2.0 * np.pi, 48)
    sphere_v = np.linspace(0.0, np.pi, 24)
    sphere_x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    sphere_y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    sphere_z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    marker_value = np.arange(point_num) if marker_color is None else marker_color
    marker_kwargs = dict(
        size=marker_size,
        opacity=marker_opacity,
        color=marker_value,
    )
    if not isinstance(marker_value, str):
        marker_kwargs["colorscale"] = color_scale

    created_new_figure = fig is None
    if fig is None:
        fig = go.Figure()

    if show_sphere and created_new_figure:
        fig.add_trace(
            go.Surface(
                x=sphere_x,
                y=sphere_y,
                z=sphere_z,
                opacity=sphere_opacity,
                showscale=False,
                colorscale=[[0.0, "#C9D6DF"], [1.0, "#52616B"]],
                hoverinfo="skip",
                name="Unit sphere",
            )
        )
        axis_colors = {"x": "#D7263D", "y": "#2E8B57", "z": "#1F77B4"}
        axis_vectors = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        for axis_name, axis_vector in axis_vectors.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, axis_vector[0]],
                    y=[0.0, axis_vector[1]],
                    z=[0.0, axis_vector[2]],
                    mode="lines",
                    line=dict(color=axis_colors[axis_name], width=6),
                    name=f"{axis_name}-axis",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=marker_kwargs,
            name=label,
        )
    )
    fig.update_layout(
        title=title or f"Sampling Points ({point_num} points)",
        width=width,
        height=height,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
            camera=dict(eye=dict(x=zoom, y=zoom, z=zoom)),
        ),
    )
    return fig


def _validate_axis_spec(axis_spec: str):
    if len(axis_spec) != 3:
        raise ValueError(f"axis_spec must have length 3, got {axis_spec!r}")
    if not axis_spec.isupper():
        raise ValueError(
            f"axis_spec must use intrinsic/body-frame axes in uppercase, got {axis_spec!r}"
        )
    for axis in axis_spec:
        if axis not in _AXIS_TO_VECTOR:
            raise ValueError(f"Unsupported axis {axis!r} in axis_spec={axis_spec!r}")


def _normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError(
                f"1D points input must have shape (3,), got {points.shape}"
            )
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    norms = np.linalg.norm(points, axis=1)
    if np.any(np.isclose(norms, 0.0)):
        raise ValueError("points contains zero-length vectors")
    return points / norms[:, None]


def _normalize_euler_angles(euler_angles: np.ndarray) -> np.ndarray:
    euler_angles = np.asarray(euler_angles, dtype=float)
    if euler_angles.ndim == 1:
        if euler_angles.shape[0] != 3:
            raise ValueError(
                f"1D euler_angles input must have shape (3,), got {euler_angles.shape}"
            )
        euler_angles = euler_angles.reshape(1, 3)
    if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
        raise ValueError(
            f"euler_angles must have shape (N, 3), got {euler_angles.shape}"
        )
    return euler_angles


def sphere_to_euler(
    axis_spec: str, points: np.ndarray, degrees: bool = False
) -> np.ndarray:
    """
    Convert sphere points to intrinsic Euler angles with the last angle fixed to 0.

    The last axis in ``axis_spec`` is treated as the object's local ``z`` axis.
    For each target point, this function computes Euler angles
    ``[a, b, 0]`` such that::

        R.from_euler(axis_spec, [a, b, 0]).apply(local_z) == point

    where ``local_z`` is the canonical basis vector of ``axis_spec[-1]``.

    Only intrinsic/body-frame Euler sequences are supported, so ``axis_spec``
    must be uppercase.

    Parameters
    ----------
    axis_spec : str
        Three-axis intrinsic Euler sequence, such as ``"ZXZ"`` or ``"XYZ"``.
    points : np.ndarray
        Sphere points with shape ``(N, 3)`` or ``(3,)``.
    degrees : bool
        If True, return Euler angles in degrees. Otherwise return radians.

    Returns
    -------
    np.ndarray
        Euler angles with shape ``(N, 3)``. The third column is always 0.
    """
    _validate_axis_spec(axis_spec)
    points = _normalize_points(points)

    axis_a = _AXIS_TO_VECTOR[axis_spec[0]]
    axis_b = _AXIS_TO_VECTOR[axis_spec[1]]
    axis_c = _AXIS_TO_VECTOR[axis_spec[2]]
    cross_bc = np.cross(axis_b, axis_c)

    euler_angles = np.zeros((len(points), 3), dtype=float)
    atol = 1e-12

    for index, point in enumerate(points):
        point_along_a = float(np.clip(np.dot(point, axis_a), -1.0, 1.0))
        alpha = float(np.dot(axis_c, axis_a))
        beta = float(np.dot(cross_bc, axis_a))

        if np.isclose(beta, 0.0, atol=atol):
            cos_b = point_along_a / alpha
            sin_b = np.sqrt(np.clip(1.0 - cos_b * cos_b, 0.0, 1.0))
            b = np.arctan2(sin_b, np.clip(cos_b, -1.0, 1.0))
        else:
            sin_b = np.clip(point_along_a / beta, -1.0, 1.0)
            cos_b = np.sqrt(np.clip(1.0 - sin_b * sin_b, 0.0, 1.0))
            # Prefer the principal solution returned by SciPy's Euler conventions.
            b = np.arctan2(sin_b, cos_b)

        rotated_local_z = R.from_rotvec(axis_b * b).apply(axis_c)
        rotated_local_z_perp = (
            rotated_local_z - np.dot(rotated_local_z, axis_a) * axis_a
        )
        point_perp = point - np.dot(point, axis_a) * axis_a

        if (
            np.linalg.norm(rotated_local_z_perp) < atol
            or np.linalg.norm(point_perp) < atol
        ):
            a = 0.0
        else:
            basis_u = rotated_local_z_perp / np.linalg.norm(rotated_local_z_perp)
            basis_v = np.cross(axis_a, basis_u)
            a = np.arctan2(np.dot(point_perp, basis_v), np.dot(point_perp, basis_u))

        euler_angles[index, 0] = a
        euler_angles[index, 1] = b

    if degrees:
        euler_angles = np.rad2deg(euler_angles)
    return euler_angles


def euler_to_sphere(
    axis_spec: str, euler_angles: np.ndarray, degrees: bool = False
) -> np.ndarray:
    """
    Convert intrinsic Euler angles to unit-sphere points.

    The last axis in ``axis_spec`` is treated as the object's local ``z`` axis.
    This function applies each Euler rotation to the canonical basis vector of
    ``axis_spec[-1]`` and returns the rotated direction on the unit sphere.

    Parameters
    ----------
    axis_spec : str
        Three-axis intrinsic Euler sequence, such as ``"ZXZ"`` or ``"XYZ"``.
    euler_angles : np.ndarray
        Euler angles with shape ``(N, 3)`` or ``(3,)``.
    degrees : bool
        If True, interpret ``euler_angles`` in degrees. Otherwise radians.

    Returns
    -------
    np.ndarray
        Sphere points with shape ``(N, 3)`` on the unit sphere.
    """
    _validate_axis_spec(axis_spec)
    euler_angles = _normalize_euler_angles(euler_angles)

    local_z = _AXIS_TO_VECTOR[axis_spec[-1]]
    points = R.from_euler(axis_spec, euler_angles, degrees=degrees).apply(local_z)
    return _normalize_points(points)
