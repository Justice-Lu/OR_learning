import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, gaussian_kde
from scipy.interpolate import UnivariateSpline
from scipy import stats
from matplotlib.colors import LogNorm

import color_function as cf 

# import logomaker

def _plotly_blank_style(fig): 
    """
    Simply update the plotly go figures to transparent background for better 3D visualization
    """
    
    fig.update_layout(
        scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False)
                ),
        margin=dict(r=10, l=10, b=10, t=10)
    )
    return fig

def _plotly_fixed_axes_ranges(fig: go.Figure, padding=0, percentage_padding=0.1, fixedrange=False) -> go.Figure:
    """
    Fix the axis ranges of a Plotly go.Figure so that hiding traces 
    does not resize the plot axes, with options for absolute or percentage-based padding.

    Parameters:
    - fig (go.Figure): A Plotly figure.
    - padding (float): Absolute padding to add/subtract from min/max axis limits.
    - percentage_padding (float): Percentage padding applied to min/max as a multiplier.

    Returns:
    - go.Figure: The modified figure with fixed axis ranges.
    """
    # Get all x and y data from traces
    x_data, y_data = [], []

    for trace in fig.data:
        if 'x' in trace and trace.x is not None:
            x_data.extend(trace.x)
        if 'y' in trace and trace.y is not None:
            y_data.extend(trace.y)

    # Determine the axis ranges
    if x_data:
        x_min, x_max = min(x_data), max(x_data)
        x_range_span = x_max - x_min  # Total span of x values
        x_padding = (x_range_span * percentage_padding) / 2  # Apply percentage padding to both sides
        x_range = [x_min - padding - x_padding, x_max + padding + x_padding]
    else:
        x_range = None

    if y_data:
        y_min, y_max = min(y_data), max(y_data)
        y_range_span = y_max - y_min  # Total span of y values
        y_padding = (y_range_span * percentage_padding) / 2  # Apply percentage padding to both sides
        y_range = [y_min - padding - y_padding, y_max + padding + y_padding]
    else:
        y_range = None

    # Update figure layout with fixed ranges
    fig.update_layout(
        xaxis=dict(range=x_range, fixedrange=fixedrange) if x_range else {},
        yaxis=dict(range=y_range, fixedrange=fixedrange) if y_range else {}
    )

    return fig

import numpy as np
import plotly.graph_objects as go

def plot_coordinates(coordinate_sets, labels=None, colors=None, opacity=0.8, size=5, 
                     mode='markers'):
    """
    Plots one or multiple sets of 3D coordinates in an interactive 3D scatter plot.

    :param coordinate_sets: List of numpy arrays, where each array has shape (N, 3) representing (x, y, z) points.
    :param labels: Optional list of labels for each coordinate set.
    :param colors: Optional list of colors corresponding to each coordinate set.
    :param opacity: Opacity of the markers (default 0.8).
    :param size: Size of the markers (default 5).
    """
    fig = go.Figure()
    
    if not isinstance(coordinate_sets, list):
        coordinate_sets = [coordinate_sets]  # Convert single input into a list
    
    num_sets = len(coordinate_sets)
    
    if labels is None:
        labels = [f"Set {i+1}" for i in range(num_sets)]
    
    if colors is None:
        colors = ["blue", "red", "green", "orange", "purple"] * (num_sets // 5 + 1)  # Cycle colors
    elif type(colors) == dict: 
        colors = [colors[_key] for _key in colors]

    for i, coords in enumerate(coordinate_sets):
        coords = np.array(coords)  # Ensure it's a numpy array
        if coords.shape[1] != 3:
            raise ValueError(f"Each coordinate set must have shape (N, 3), but got {coords.shape}")

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode=mode,
            name=labels[i],
            marker=dict(
                size    = size[i]    if isinstance(size, list)    else size,
                color   = colors[i]  if isinstance(colors, list)  else colors,
                opacity = opacity[i] if isinstance(opacity, list) else opacity
            )
        ))

    # Set layout
    fig = _plotly_blank_style(fig)

    return fig

    
# def visualize_voxel_grid(voxel_data, 
#                          labels, 
#                          colormap, 
#                          size=5,
#                          opacity=0.1,
#                          highlight_labels=None, 
#                          highlight_opacity=0.5):
#     """
#     Visualizes 3D voxel grids using Plotly Scatter3D.

#     :param voxel_data: 
#         List of voxel grids. Each grid is a numpy array of shape (X, Y, Z).
#     :param coordinate_labels: 
#         Dictionary where keys are labels (e.g., "BACKBONE", "BOUND_POINT") and values are voxel grids.
#     :param color_map: 
#         Dictionary mapping labels to colors for visualization.
#     :param highlight_labels: 
#         List of labels to highlight with higher opacity. Default is None.
#     """
#     fig = go.Figure()

#     for i, (label, voxel_grid) in enumerate(zip(labels, voxel_data)):
#         # Get occupied voxels
#         occupied_voxels = np.array(np.where(voxel_grid != 0)).T
        
#         if len(occupied_voxels) == 0:
#             continue  # Skip empty grids
        
#         # Extract x, y, z coordinates
#         x = occupied_voxels[:, 0]
#         y = occupied_voxels[:, 1]
#         z = occupied_voxels[:, 2]
        
#         # Add scatter plot for the current label
#         fig.add_trace(go.Scatter3d(
#             x=x, y=y, z=z,
#             mode='markers',
#             name=label,
#             marker=dict(
#                 size=size[i] if isinstance(size, list) else size,
#                 color=colormap.get(label, 'gray') if type(colormap) == dict else colormap[i],  # Default color is gray if label is not in the color_map
#                 opacity= highlight_opacity if highlight_labels and label in highlight_labels else opacity
#             )
#         ))
    
#     # Apply a blank style 
#     fig = _plotly_blank_style(fig)
    
#     return fig

def visualize_voxel_grid(voxel_data, 
                         labels=None, 
                         colormap=None, 
                         size=5,
                         opacity=0.1,
                         voxel_type='4D', 
                         highlight_labels=None, 
                         highlight_opacity=0.5,
                         property_indices=None):
    """
    Visualizes 3D voxel grids using Plotly Scatter3D. Works with both 3D and 4D voxel representations.

    Parameters:
    -----------
    voxel_data : list or array
        For voxel_type='3D': List of voxel grids, each with shape (X, Y, Z)
        For voxel_type='4D': List of voxel grids, each with shape (X, Y, Z, P) where P is properties
        For single voxel visualization, you can pass a single array
    labels : list
        List of labels corresponding to each voxel grid for legend display
        If None, will generate labels like "Voxel 1", "Voxel 2", etc.
    colormap : dict or list
        Dictionary mapping labels to colors, or a list of colors for each voxel grid
        If None, will use default Plotly colors
    size : int or list
        Size of markers. Can be a single value or a list for different sizes per voxel grid
    opacity : float
        Opacity of markers
    voxel_type : str
        '3D' - Properties encoded as integers in Z layer (shape: X, Y, Z)
        '4D' - One-hot encoded properties (shape: X, Y, Z, P)
    highlight_labels : list
        List of labels to highlight with higher opacity
    highlight_opacity : float
        Opacity for highlighted voxels
    property_indices : list or None
        For 4D voxels: Which property indices to visualize (e.g., [0, 3, 5])
        For 3D voxels: Which property values to visualize (e.g., [1, 3, 5])
        If None:
            - Will visualize all properties combined in a single trace per voxel (without property-specific labeling)
            - For 3D voxels: Will include all values except -1 (negative space)
            - For 4D voxels: Will include any non-zero property values

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Plotly figure object for the visualization
    """
    fig = go.Figure()
    
    # Ensure voxel_data is a np.array
    if not isinstance(voxel_data, type(np.array)):
        voxel_data = np.array(voxel_data)
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f"Voxel {i+1}" for i in range(len(voxel_data))]
    
    # Generate default colormap if not provided
    if colormap is None:
        # Default plotly colorscale
        colormap = cf.distinct_colors(labels)
        # colormap = {label: colors[i % len(colors)] for i, label in enumerate(labels)}
    
    # Process each voxel grid
    for i, voxel_grid in enumerate(voxel_data):
        label = labels[i] if i < len(labels) else f"Voxel {i+1}"
        
        if voxel_type == '3D':
            if property_indices is None:
                # Combined mode: plot all valid properties together
                # Get all voxels except negative space (-1)
                occupied_voxels = np.array(np.where(voxel_grid > -1)).T
                
                if len(occupied_voxels) == 0:
                    continue  # Skip if no valid voxels
                
                # Extract coordinates
                x, y, z = occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]
                
                # Determine color
                if isinstance(colormap, dict):
                    color = colormap.get(label, 'gray')
                else:
                    color = colormap[i % len(colormap)]
                
                # Determine if this should be highlighted
                is_highlighted = highlight_labels and label in highlight_labels
                
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    name=label,
                    marker=dict(
                        size=size[i] if isinstance(size, list) else size,
                        color=color,
                        opacity=highlight_opacity if is_highlighted else opacity
                    )
                ))
            else:
                # Original mode: plot each property separately
                for prop_idx in property_indices:
                    # Get voxels with this property value
                    occupied_voxels = np.array(np.where(voxel_grid == prop_idx)).T
                    
                    if len(occupied_voxels) == 0:
                        continue  # Skip if no voxels have this property
                    
                    # Extract coordinates
                    x, y, z = occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]
                    
                    # Create label for this property
                    prop_label = f"{label} - Property {prop_idx}"
                    
                    # Determine color - if colormap is a dict, try to use label, then prop_label
                    if isinstance(colormap, dict):
                        color = colormap.get(prop_label, 
                               colormap.get(label, 
                               colormap.get(prop_idx, 'gray')))
                    else:
                        # If colormap is a list, use the corresponding color
                        color = colormap[i % len(colormap)]
                    
                    # Determine if this should be highlighted
                    is_highlighted = highlight_labels and (label in highlight_labels or prop_label in highlight_labels)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        name=prop_label,
                        marker=dict(
                            size=size[i] if isinstance(size, list) else size,
                            color=color,
                            opacity=highlight_opacity if is_highlighted else opacity
                        )
                    ))
                
        elif voxel_type == '4D':
            # 4D voxels - one-hot encoded properties
            if len(voxel_grid.shape) != 4:
                raise ValueError(f"Expected 4D voxel with shape (X,Y,Z,P), got {voxel_grid.shape}")
            
            if property_indices is None:
                # Combined mode: Create a single trace with all non-zero properties
                # First identify all occupied voxels (where any property is non-zero)
                # Sum across all properties
                any_property = np.sum(voxel_grid, axis=3) > 0
                occupied_voxels = np.array(np.where(any_property)).T
                
                if len(occupied_voxels) == 0:
                    continue  # Skip if no valid voxels
                
                # Extract coordinates
                x, y, z = occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]
                
                # Determine color
                if isinstance(colormap, dict):
                    color = colormap.get(label, 'gray')
                else:
                    color = colormap[i % len(colormap)]
                
                # Determine if this should be highlighted
                is_highlighted = highlight_labels and label in highlight_labels
                
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    name=label,
                    marker=dict(
                        size=size[i] if isinstance(size, list) else size,
                        color=color,
                        opacity=highlight_opacity if is_highlighted else opacity
                    )
                ))
            else:
                # Original mode: plot each property separately
                for p_idx in property_indices:
                    # Extract the 3D grid for this property
                    prop_grid = voxel_grid[:, :, :, p_idx]
                    
                    # Get occupied voxels (non-zero values)
                    occupied_voxels = np.array(np.where(prop_grid > 0)).T
                    
                    if len(occupied_voxels) == 0:
                        continue  # Skip if no voxels have this property
                    
                    # Extract coordinates
                    x, y, z = occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]
                    
                    # Create label for this property
                    prop_label = f"{label} - Property {p_idx}"
                    
                    # Determine color
                    if isinstance(colormap, dict):
                        color = colormap.get(prop_label, 
                               colormap.get(label, 
                               colormap.get(p_idx, 'gray')))
                    else:
                        color = colormap[i % len(colormap)]
                    
                    # Determine if this should be highlighted
                    is_highlighted = highlight_labels and (label in highlight_labels or prop_label in highlight_labels)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        name=prop_label,
                        marker=dict(
                            size=size[i] if isinstance(size, list) else size,
                            color=color,
                            opacity=highlight_opacity if is_highlighted else opacity
                        )
                    ))
        else:
            raise ValueError(f"Unknown voxel_type: {voxel_type}. Must be '3D' or '4D'")
    
    # Apply a blank style
    fig = _plotly_blank_style(fig)
    return fig

from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

def plt_voxel_slices(
        protein_points,
        protein_res,
        imp_points,
        imp_values,
        imp_max=None, imp_min=None, 
        angles=[45, 135],
        slice_axis="z",
        slice_thickness=5,
        n_slices=1,
        slice_position=None, 
        cmap="RdBu_r",
        protein_color="lightgray",
        protein_size=30, protein_alpha=0.3, 
        imp_size=20,
        label_residues=False,
        show_slice_planes=True
    ):
    """
        Visualize rotated orthogonal slices of a protein point cloud and corresponding
        importance/voxel values in both 3D and 2D.

        This function produces two separate figures:
        
        1. **fig3d** — A figure containing one 3D subplot per rotation angle.
        Each subplot shows the rotated protein point cloud and (optionally)
        the slice planes indicating where 2D projections will be taken.

        2. **fig2d** — A figure containing `len(angles) × n_slices` 2D subplots.
        Each row corresponds to a rotation angle, and each column corresponds
        to a slice position.  
        Every subplot shows:
        - The projected protein points inside the slice
        - The projected importance/voxel points
        - (Optional) labels for protein residues, plotted once per unique residue
        - (Optional) smooth convex-hull outlines for each residue cluster

        Parameters
        ----------
        protein_points : (N, 3) array
            Cartesian coordinates of protein atoms/points.

        protein_res : (N,) array
            Residue index or residue label for each protein point.
            Used for grouping and labeling residue clusters within slices.

        imp_points : (M, 3) array
            Cartesian coordinates of importance/voxel centers.

        imp_values : (M,) array
            Scalar importance values associated with each `imp_point`.
            Used for colormap visualization.

        angles : list of float, optional
            Z-rotation angles (in degrees) at which to generate slices.
            Each angle produces one row of 3D/2D slices.

        slice_axis : {"x", "y", "z"}, optional
            Axis along which slices are taken *after* rotation.
            Defines which coordinate is compared to `slice_center ± slice_thickness/2`.

        slice_thickness : float, optional
            Thickness of each slice (in coordinate units).  
            Points satisfy:  
            `abs(coord[axis] - slice_center) <= slice_thickness / 2`.

            Ignored if `slice_position` is provided, except for determining
            how *wide* each slice is.

        n_slices : int, optional
            Number of slices per angle *if* `slice_position` is not provided.

            If `slice_position` **is** provided, `n_slices` is automatically
            set to `len(slice_position)`.

        slice_position : list of float in [0, 1], optional
            Normalized slice locations along the selected `slice_axis`.

            If provided:
            - `n_slices = len(slice_position)`
            - Slice centers are computed as:

            ``slice_center = amin + pos * (amax - amin)``

            where `amin` and `amax` are the min/max of the protein along
            the axis *after* rotation.

            This allows slices to be placed at biologically meaningful
            percentiles (e.g., [0.10, 0.50, 0.90]) instead of uniform spacing.

        cmap : str, optional
            Colormap used to visualize `imp_values`.

        protein_color : matplotlib color, optional
            Color used for protein scatter points in 2D slices.

        protein_size : float, optional
            Marker size for protein scatter points.

        imp_size : float, optional
            Marker size for importance/voxel scatter points.

        label_residues : bool, optional
            If True:
            - Each residue is labeled once per slice (never per point).
            - Label is positioned outside the residue’s convex hull.
            - A smooth outline of the convex hull is drawn using spline smoothing.

        show_slice_planes : bool, optional
            If True, draws semi-transparent slice planes inside each 3D plot.

        Returns
        -------
        fig3d : matplotlib.figure.Figure
            Figure containing the 3D views, one per rotation angle.

        axes3d : list of Axes3D
            List of 3D Axes corresponding to each angle.

        fig2d : matplotlib.figure.Figure
            Figure containing the 2D slices in a grid layout
            (`len(angles)` rows × `n_slices` columns).

        axes2d : 2D list of Axes
            `axes2d[i][j]` is the j-th slice for the i-th rotation angle.

        Notes
        -----
        - All coordinates are first centered by subtracting the mean of
        `protein_points`. This keeps the protein visually stable across rotations.
        - Slice positions apply *after* rotation, ensuring intuitive behavior
        even for non-axis-aligned structures.
        - Residue labeling uses:
            - Clustered coordinates per residue
            - Mean positions of convex hull for label placement
            - Spline-smoothed convex hulls for clean outlines
        - Suitable for visualizing voxelized features, attention maps,
        CNN importance fields, and cavity slices.

        Examples
        --------
        >>> fig3d, axes3d, fig2d, axes2d = plt_voxel_slices(
        ...     protein_points, protein_res,
        ...     imp_points, imp_values,
        ...     angles=[0, 45, 90],
        ...     slice_axis="z",
        ...     slice_thickness=4,
        ...     n_slices=3
        ... )

        >>> # Using customized slice positions
        >>> fig3d, axes3d, fig2d, axes2d = plt_voxel_slices(
        ...     protein_points, protein_res,
        ...     imp_points, imp_values,
        ...     slice_axis="y",
        ...     slice_position=[0.1, 0.5, 0.9]
        ... )
    """

    # -----------------------------
    # Center coordinates
    # -----------------------------
    center = protein_points.mean(axis=0)
    prot = protein_points - center
    imp  = imp_points - center

    # Bounds for full subplot extents
    xmin, xmax = prot[:,0].min(), prot[:,0].max()
    ymin, ymax = prot[:,1].min(), prot[:,1].max()
    zmin, zmax = prot[:,2].min(), prot[:,2].max()

    axis_map = {"x":0, "y":1, "z":2}
    axis_i = axis_map[slice_axis]

    vmax = np.max(np.abs(imp_values))

    if imp_min: vmin = imp_min
    if imp_max: vmax = imp_max
    elif (imp_max is None) & (imp_min is None): 
        vmin = -vmax

    # -----------------------------
    # Create two separate figures
    # -----------------------------
    fig3d, axes3d = plt.subplots(len(angles), 1,
                                 subplot_kw={"projection": "3d"},
                                 figsize=(4, 3*len(angles)))

    if len(angles) == 1:
        axes3d = [axes3d]


    if slice_position: # Prioritize slice position instead
        n_slices = len(slice_position)
    fig2d, axes2d = plt.subplots(len(angles), n_slices,
                                 figsize=(3*n_slices, 3*len(angles)))

    if len(angles) == 1:
        axes2d = [axes2d]

    # -----------------------------
    # Loop over angles
    # -----------------------------
    for ai, angle in enumerate(angles):
        rot = R.from_euler("z", angle, degrees=True)
        prot_rot = rot.apply(prot)
        imp_rot  = rot.apply(imp)
        
        # -----------------------------
        # Compute slice centers
        # -----------------------------
        amin = prot_rot[:, axis_i].min()
        amax = prot_rot[:, axis_i].max()

        if slice_position is not None:
            # User-defined normalized slice locations (0→1)
            slice_position = np.asarray(slice_position, dtype=float)
            slice_position = np.clip(slice_position, 0.0, 1.0)
            n_slices = len(slice_position)

        else:
            # Auto-generate normalized slice positions
            # EVENLY spaced across [0, 1]
            slice_position = np.linspace(0.0, 1.0, n_slices)

        # Physical slice centers (coordinate units)
        slice_centers = amin + slice_position * (amax - amin)


        # -----------------------------
        # Plot 3D view (first column)
        # -----------------------------
        ax3d = axes3d[ai]

        # Protein stays fixed
        ax3d.scatter(prot[:,0], prot[:,1], prot[:,2],
                    c="gray", s=3, alpha=0.05)

        # Importance values (rotated!)
        ax3d.scatter(
            imp[:,0], imp[:,1], imp[:,2],
            c=imp_values,
            cmap=cmap,
            s=5,
            vmin=vmin,
            vmax=vmax,
            alpha=0.9
        )

        # Set limits based ONLY on protein (keeps view stable)
        ax3d.set_xlim(xmin, xmax)
        ax3d.set_ylim(ymin, ymax)
        ax3d.set_zlim(zmin, zmax)

        # Show slice planes
        if show_slice_planes:
            plane_rot = R.from_euler("z", angle, degrees=True)
            plane_res = 2   # corners only, for transparency and speed

            for c in slice_centers:
                if slice_axis == "x":
                    Y, Z = np.meshgrid(np.linspace(ymin,ymax,plane_res),
                                    np.linspace(zmin,zmax,plane_res))
                    X = np.ones_like(Y)*c
                elif slice_axis == "y":
                    X, Z = np.meshgrid(np.linspace(xmin,xmax,plane_res),
                                    np.linspace(zmin,zmax,plane_res))
                    Y = np.ones_like(X)*c
                else:
                    X, Y = np.meshgrid(np.linspace(xmin,xmax,plane_res),
                                    np.linspace(ymin,ymax,plane_res))
                    Z = np.ones_like(X)*c

                pts = np.column_stack([X.ravel(),Y.ravel(),Z.ravel()])
                pts_rot = plane_rot.apply(pts)
                Xr, Yr, Zr = [x.reshape(plane_res,plane_res) for x in pts_rot.T]

                ax3d.plot_surface(Xr, Yr, Zr, color="gray", alpha=0.05)

        # Clean 3D plot
        ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
        ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
        ax3d.axis("off")
        ax3d.set_title(f"Slice Angle {angle}°", fontsize=10)

        # -----------------------------
        # 2D Slice Views (Fig 2)
        # -----------------------------
        for si, c in enumerate(slice_centers):
            ax = axes2d[ai][si] if n_slices > 1 else axes2d[ai]

            low = c - slice_thickness/2
            high = c + slice_thickness/2

            prot_mask = (prot_rot[:,axis_i] >= low) & (prot_rot[:,axis_i] < high)
            imp_mask  = (imp_rot[:,axis_i]  >= low) & (imp_rot[:,axis_i]  < high)

            prot_slice = prot_rot[prot_mask]
            imp_slice  = imp_rot[imp_mask]
            imp_vals   = imp_values[imp_mask]

            # Coordinate projection
            if slice_axis == "x":
                Xp, Yp = prot_slice[:,1], prot_slice[:,2]
                Xi, Yi = imp_slice[:,1], imp_slice[:,2]
            elif slice_axis == "y":
                Xp, Yp = prot_slice[:,0], prot_slice[:,2]
                Xi, Yi = imp_slice[:,0], imp_slice[:,2]
            else: # z
                Xp, Yp = prot_slice[:,0], prot_slice[:,1]
                Xi, Yi = imp_slice[:,0], imp_slice[:,1]

            # Protein scatter
            if len(prot_slice) > 0:
                ax.scatter(Xp, Yp, c=protein_color, s=protein_size, alpha=protein_alpha)

                # Residue labeling + convex hull outlines
                if label_residues:
                    res_labels = protein_res[prot_mask]
                    for lab in np.unique(res_labels):
                        if lab == 'None': # Skip for if the label didn't match to anything 
                            continue
                        mask = res_labels == lab
                        x_lab = Xp[mask]; y_lab = Yp[mask]

                        if len(x_lab) > 5:
                            pts = np.column_stack([x_lab,y_lab])
                            hull = ConvexHull(pts)
                            hull_pts = pts[hull.vertices]

                            hull_pts = np.vstack([hull_pts, hull_pts[0]])
                            tck, u = splprep([hull_pts[:,0], hull_pts[:,1]], 
                                             s=1.0, per=True)
                            uu = np.linspace(0,1,200)
                            xs, ys = splev(uu, tck)
                            ax.plot(xs, ys, 
                                    c="black", 
                                    lw=2, 
                                    alpha=0.1)

                            # label offset outward
                            xm, ym = hull_pts[:,0].mean(), hull_pts[:,1].mean()
                            ax.text(xm, ym, str(lab),
                                    fontsize=10, ha="center", va="center")

            # Importance scatter
            if len(imp_slice) > 0:
                ax.scatter(Xi, Yi, c=imp_vals, cmap=cmap, s=imp_size,
                           vmin=vmin, vmax=vmax)

            # Dynamic bounds AFTER rotation
            xmin_r, xmax_r = prot_rot[:,0].min(), prot_rot[:,0].max()
            ymin_r, ymax_r = prot_rot[:,1].min(), prot_rot[:,1].max()
            zmin_r, zmax_r = prot_rot[:,2].min(), prot_rot[:,2].max()

            pad = 2  # padding so labels + hulls fit
            if slice_axis == "x": ax.set_xlim(ymin_r - pad, ymax_r + pad); ax.set_ylim(zmin_r - pad, zmax_r + pad)
            elif slice_axis == "y": ax.set_xlim(xmin_r - pad, xmax_r + pad); ax.set_ylim(zmin_r - pad, zmax_r + pad)
            else: ax.set_xlim(xmin_r - pad, xmax_r + pad) ; ax.set_ylim(ymin_r - pad, ymax_r + pad)
            
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_aspect("equal")
            ax.set_title(f"Slice Angle: {angle}°, Position: {slice_position[si]}", fontsize=9)
    
    fig3d.tight_layout()
    fig2d.tight_layout()

    return fig3d, axes3d, fig2d, axes2d



def plot_correlation(df, x_by, y_by, 
                     label_by=None, 
                     xlabel='', ylabel='', title='', 
                     color_by=None, 
                     size=10, showlegend=True, 
                     plot_pearson_line=True, 
                     linestyle='dash', linecolor=None,
                     opacity=0.7, text_xy=[0.05, 0.95], 
                     figsize=(800, 800), 
                     padding=0.05):
    """
    Creates an interactive scatter plot with data grouped by a categorical column (e.g., `cid`).

    :param df: Pandas DataFrame containing the data.
    :param x_by: Column name for x-axis values.
    :param y_by: Column name for y-axis values.
    :param label_by: Column for hover text (optional).
    :param xlabel: String, label for the x-axis.
    :param ylabel: String, label for the y-axis.
    :param title: String, title of the plot.
    :param color_by: Column to color traces by (optional). Defaults to gray if not provided.
    :param size: Size of scatter points.
    :param showlegend: Whether to show legend entries.
    :param plot_pearson_line: Whether to show Pearson correlation line.
    :return: A Plotly figure.
    """

    # Extract relevant columns
    values1 = df[x_by].values
    values2 = df[y_by].values
    labels = df[label_by].values if label_by else [f"({x:.2f}, {y:.2f})" for x, y in zip(values1, values2)]  # Default hover text
    groups = df[color_by].unique() if color_by else ['All Data']  # Single group if color_by is missing

    # Generate distinct colors or default to gray
    if color_by:
        color_map = cf.distinct_colors(groups, category= 'tab10' if len(groups) < 10 else 'tab20')
    else:
        color_map = {'All Data': 'gray'}

    fig = go.Figure()

    # Loop over each unique group and add separate traces
    for group in groups:
        subset = df[df[color_by] == group] if color_by else df  # Use full data if no color_by
        
        hover_text = [
            f"{x_by}: {x:.3f}<br>{y_by}: {y:.3f}<br>{label}" 
            for x, y, label in zip(subset[x_by], subset[y_by], subset[label_by] if label_by else [''])
        ]
        
        fig.add_trace(go.Scatter(
            x=subset[x_by], 
            y=subset[y_by], 
            mode='markers',
            marker=dict(color=color_map[group], size=size, opacity=opacity),
            name=f"{group}" if color_by else "Data",  # Legend entry
            # text=subset[label_by] if label_by else [f"({x:.2f}, {y:.2f})" for x, y in zip(subset[x_by], subset[y_by])],
            text=hover_text,
            hoverinfo="text",
            legendgroup=f"{group}" if color_by else "Data",  # Group legend items
            showlegend=showlegend if color_by else False  # Hide legend if no grouping
        ))

    # Calculate Pearson correlation and best-fit line
    r, p_value = pearsonr(values1, values2)
    slope, intercept, _, _, _ = linregress(values1, values2)

    # Add Pearson correlation line
    if plot_pearson_line:
        sorted_idx = np.argsort(values1)
        sorted_x = values1[sorted_idx]
        sorted_y = slope * sorted_x + intercept

        fig.add_trace(go.Scatter(
            x=sorted_x, y=sorted_y, mode='lines',
            line=dict(color=linecolor if linecolor else 'black' if p_value < 0.05 else 'gray', 
                      dash=linestyle),
            text=[f"x: {x:.2f}<br>y: {y:.2f}<br>Pearson r = {r:.3f}<br>r² = {r**2:.3f}<br>p-value = {p_value:.3e}" for x, y in zip(sorted_x, sorted_y)],
            hoverinfo="text",
            name="Pearson Line",
            showlegend=showlegend
        ))

    # Add Pearson correlation text annotation
    fig.add_annotation(
        x=text_xy[0], y=text_xy[1], 
        text=f"Pearson r = {r:.3f}<br>r² = {r**2:.3f}<br>p-value = {p_value:.3e}",
        showarrow=False,
        xref="paper", yref="paper",
        # bgcolor="white", 
        # bordercolor="gray",
        opacity=0.8
    )

    # Compute axis limits with padding
    x_min, x_max = values1.min(), values1.max()
    y_min, y_max = values2.min(), values2.max()
    x_pad = padding * (x_max - x_min)
    y_pad = padding * (y_max - y_min)

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or x_by,
        yaxis_title=ylabel or y_by,
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),  # Fixed range with padding
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),  # Fixed range with padding
        template="simple_white"
    )

    if figsize: 
        fig.update_layout(width=figsize[0], height=figsize[1])

    return fig


def plt_correlation(values1, values2,
                    xlabel='', 
                    ylabel='',
                    title='', 
                    plot_pearson_line=True, 
                    linestyle='dotted',
                    linecolor='Red',
                    linewidth=3, 
                    linealpha=0.5,
                    edgecolor='gray',
                    edgesize=10, 
                    opacity=0.5, 
                    text_xy=[0.05, 0.95], 
                    figsize=[5, 5],
                    colorbar=False, 
                    plot_style='scatter',  # 'scatter', 'hexbin', or 'hist2d'
                    bins=100, cmap='Greys', 
                    log_scale=True, 
                    spines=False, 
                    **kwargs):
    """
    Generate a correlation plot between two sets of values using various 2D plotting styles.

    The function supports scatter, hexbin, and 2D histogram visualizations. It also computes
    and optionally plots the Pearson correlation line, along with annotation of the correlation
    coefficient (r), r², and p-value.

    Parameters:
        values1 (array-like): Values to be plotted on the x-axis.
        values2 (array-like): Values to be plotted on the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        plot_pearson_line (bool): Whether to plot the Pearson correlation line and stats.
        linestyle (str): Line style for the Pearson correlation line.
        linecolor (str): Color of the Pearson correlation line.
        edgecolor (str): Marker color for scatter plot points.
        edgesize (float): Marker size for scatter plot points.
        opacity (float): Transparency for scatter plot markers.
        text_xy (list): (x, y) relative axes coordinates for placing the correlation text box.
        figsize (list): Size of the figure in inches [width, height].
        colorbar (bool): Whether to display a colorbar for hexbin or hist2d plots.
        plot_style (str): One of 'scatter', 'hexbin', or 'hist2d'.
        bins (int): Number of bins for hexbin or hist2d plots.
        cmap (str): Colormap used for hexbin or hist2d plots.
        log_scale (bool): Whether to apply logarithmic normalization to the color scale.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the plot.
    """

    values1 = np.array(values1)
    values2 = np.array(values2)

    # Compute Pearson stats
    r, p_value = pearsonr(values1, values2)
    slope, intercept, _, _, _ = linregress(values1, values2)

    # Create OO figure
    fig, ax = plt.subplots(figsize=figsize)
    norm = LogNorm() if log_scale else None

    if plot_style == 'scatter':
        ax.scatter(values1, values2, color=edgecolor, alpha=opacity, s=edgesize)
        
    elif plot_style == 'scatter_line': 
        smooth = kwargs.get('smooth_scatter_line', False)
        if smooth:
            degree = kwargs.get('poly_degree', 1)
            coeffs = np.polyfit(values1, values2, deg=degree)
            poly = np.poly1d(coeffs)

            x_fit = np.linspace(np.min(values1), np.max(values1), 500)
            y_fit = poly(x_fit)
            ax.plot(x_fit, y_fit, color=edgecolor, alpha=opacity, linewidth=edgesize)
        else:
            ax.plot(values1, values2, color=edgecolor, alpha=opacity, linewidth=edgesize)
    elif plot_style == 'hexbin':
        hb = ax.hexbin(values1, values2, gridsize=bins, cmap=cmap, norm=norm, mincnt=1)
        if colorbar:
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Counts (log)' if log_scale else 'Counts')

    elif plot_style == 'hist2d':
        counts, xedges, yedges, img = ax.hist2d(values1, values2, bins=bins, cmap=cmap, norm=norm)
        if colorbar:
            cb = fig.colorbar(img, ax=ax)
            cb.set_label('Counts (log)' if log_scale else 'Counts')

    else:
        raise ValueError(f"plot_style must be one of ['scatter', 'hexbin', 'hist2d'], got '{plot_style}'.")

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Regression line and annotation
    if plot_pearson_line:
        sorted_idx = np.argsort(values1)
        sorted_x = values1[sorted_idx]
        sorted_y = slope * sorted_x + intercept

        ax.plot(sorted_x, sorted_y, color=linecolor, 
                linestyle=linestyle, linewidth=linewidth, 
                alpha=linealpha)

        ax.text(
            text_xy[0], text_xy[1],
            f"Pearson r = {r:.3f}\n$r^2$ = {r**2:.3f}\np-value = {p_value:.3e}",
            fontsize=kwargs.get('pearson_fontsize', 5),
            ha="left", va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", 
                      edgecolor="white", 
                      facecolor="white", alpha=0),
        )

    fig.tight_layout()
    
    if not spines: 
        fig = plt_clean_axes(fig)
        
    return fig, ax 

def plt_correlation_subplots(values_pairs,
                    ax = None, 
                    ncols=1,
                    xlabels='', 
                    ylabels='',
                    titles='', 
                    plot_pearson_line=True, 
                    linestyle='dotted',
                    linecolor='Red',
                    linewidth=3, 
                    linealpha=0.5,
                    edgecolor='gray',
                    edgesize=10, 
                    opacity=0.5, 
                    text_xy=[0.05, 0.95], 
                    figsize_per_plot=(4,4),
                    # figsize=[8, 8],
                    colorbar=False, 
                    plot_style='scatter',  # 'scatter', 'hexbin', or 'hist2d'
                    bins=100, cmap='Greys', 
                    log_scale=True, 
                    spines=False,
                    **kwargs):
    """
    Generate a correlation plot between two sets of values using various 2D plotting styles.

    The function supports scatter, hexbin, and 2D histogram visualizations. It also computes
    and optionally plots the Pearson correlation line, along with annotation of the correlation
    coefficient (r), r², and p-value.

    Parameters:
        values1 (array-like): Values to be plotted on the x-axis.
        values2 (array-like): Values to be plotted on the y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        plot_pearson_line (bool): Whether to plot the Pearson correlation line and stats.
        linestyle (str): Line style for the Pearson correlation line.
        linecolor (str): Color of the Pearson correlation line.
        edgecolor (str): Marker color for scatter plot points.
        edgesize (float): Marker size for scatter plot points.
        opacity (float): Transparency for scatter plot markers.
        text_xy (list): (x, y) relative axes coordinates for placing the correlation text box.
        figsize (list): Size of the figure in inches [width, height].
        colorbar (bool): Whether to display a colorbar for hexbin or hist2d plots.
        plot_style (str): One of 'scatter', 'hexbin', or 'hist2d'.
        bins (int): Number of bins for hexbin or hist2d plots.
        cmap (str): Colormap used for hexbin or hist2d plots.
        log_scale (bool): Whether to apply logarithmic normalization to the color scale.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the plot.
    """

    n = len(values_pairs)
    nrows = int(np.ceil(n / ncols))
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    
    
    for i, (values1, values2) in enumerate(values_pairs):
        ax = axes[i]
        values1 = np.array(values1)
        values2 = np.array(values2)

        # Pearson
        r, p_value = pearsonr(values1, values2)
        slope, intercept, *_ = linregress(values1, values2)

        # Plot style
        style = kwargs.get('plot_style', plot_style)
        norm = LogNorm() if kwargs.get('log_scale', log_scale) else None

        if style == 'scatter':
            ax.scatter(values1, values2,
                       color=kwargs.get('edgecolor', edgecolor),
                       alpha=kwargs.get('opacity', opacity),
                       s=kwargs.get('edgesize', edgesize))
        elif style == 'hexbin':
            hb = ax.hexbin(values1, values2, gridsize=kwargs.get('bins', bins),
                           cmap=kwargs.get('cmap', cmap), norm=norm, mincnt=1)
            if kwargs.get('colorbar', colorbar):
                fig.colorbar(hb, ax=ax)
        elif style == 'hist2d':
            counts, xedges, yedges, img = ax.hist2d(values1, values2, 
                                                    # vmax=kwargs.get('hist2d_vmax', max(max(values1), max(values2))),
                                                    bins=kwargs.get('bins', bins),
                                                    cmap=kwargs.get('cmap', cmap),
                                                    norm=norm
                                                    )
            if kwargs.get('colorbar', colorbar):
                fig.colorbar(img, ax=ax)
        else:
            raise ValueError("Invalid plot_style")

        if kwargs.get('plot_pearson_line', plot_pearson_line):
            sorted_idx = np.argsort(values1)
            ax.plot(values1[sorted_idx],
                    slope * values1[sorted_idx] + intercept,
                    color=kwargs.get('linecolor', linecolor),
                    linestyle=kwargs.get('linestyle', linestyle),
                    linewidth=kwargs.get('linewidth', linewidth),
                    alpha=kwargs.get('linealpha', linealpha))
            tx, ty = kwargs.get('text_xy', text_xy)
            ax.text(tx, ty,
                    f"Pearson r = {r:.3f}\n$r^2$ = {r**2:.3f}\np = {p_value:.1e}",
                    fontsize=kwargs.get('pearson_fontsize', 5),
                    ha='left', va='top',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3",
                              edgecolor="white",
                              facecolor="white", alpha=0))

        # Labels
        if xlabels: ax.set_xlabel(xlabels[i])
        if ylabels: ax.set_ylabel(ylabels[i])
        if titles:  ax.set_title(titles[i])

        if not kwargs.get('spines', spines):
            fig = plt_clean_axes(fig)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes[:n]


def plt_clean_axes(fig, remove_spines=True, spine_color='black', tick_direction='out'):
    """
    Applies consistent styling to all axes in the figure.

    Parameters:
        fig: matplotlib Figure object
        remove_spines: bool, if True, hides top and right spines
        spine_color: str, color to apply to visible spines
        tick_direction: str, direction of ticks ('in', 'out', 'inout')
    """
    for ax in fig.axes:
        if remove_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(spine_color)
        
        ax.tick_params(direction=tick_direction, colors=spine_color)

        # Optional: remove grid or background
        ax.set_facecolor('white')
    return fig


def plot_weblogo(frequency_matrix, 
                 alphabet=list("ACDEFGHIKLMNPQRSTVWY-"), 
                 highlight_index=None, 
                 highlight_color="gold", 
                 highlight_text_color="gold", 
                 highlight_text_size=10, 
                 bw_index=None,
                 figsize=[12, 4], 
                 row_length=50, 
                 plot_title='',
                 show_axticks=True):
    """
    Plots a long WebLogo in multiple rows for better visualization, with optional highlighted positions.
    
    Parameters:
        frequency_matrix (np.ndarray): Frequency matrix (AA x Positions).
        alphabet (list): List of amino acids including gap ('-').
        highlight_index (dict): Mapping of highlight labels (e.g., "H104") to positions in the alignment.
        figsize (list): Size of the figure [width, height per row].
        row_length (int): Number of positions per row in the WebLogo.
    
    Returns:
        matplotlib.figure.Figure: The figure containing the WebLogo plot.
    """
    import matplotlib.pyplot as plt
    import logomaker

    # Identify and remove gaps ('-') from the frequency matrix and alphabet
    if '-' in alphabet:
        gap_index = alphabet.index('-')
        frequency_matrix = np.delete(frequency_matrix, gap_index, axis=0)
        alphabet = [aa for aa in alphabet if aa != '-']

    # Split frequency matrix into chunks
    num_positions = frequency_matrix.shape[1]
    num_rows = (num_positions + row_length - 1) // row_length  # Compute the number of rows

    # Initialize the figure
    fig = plt.figure(figsize=(figsize[0], figsize[1] * num_rows))

    for i in range(num_rows):
        start = i * row_length
        end = min((i + 1) * row_length, num_positions)

        # Extract the chunk for the current row
        chunk = frequency_matrix[:, start:end]
        frequency_df = pd.DataFrame(chunk, index=alphabet)

        # Create subplot for each row
        ax = plt.subplot(num_rows, 1, i + 1)
        logo = logomaker.Logo(
            frequency_df.T,
            ax=ax,
            shade_below=0.5,
            fade_below=0.5
        )

        # Highlight specified positions within the current chunk
        if highlight_index:
            for label, global_position in highlight_index.items():
                global_position -= 1
                if start <= global_position < end:
                    local_position = global_position - start
                    logo.highlight_position(p=local_position, color=highlight_color, alpha=0.5)
                    ax.text(local_position, -0.1, label, color=highlight_text_color, 
                            ha="center", fontsize=highlight_text_size, rotation=0)

        if bw_index:
            for label, global_position in bw_index.items():
                global_position -= 1
                if start <= global_position < end:
                    local_position = global_position - start
                    logo.highlight_position(p=local_position, color='gray', alpha=0.5)
                    ax.text(local_position, -0.05, label, color="gray", ha="center", fontsize=10, rotation=0)

        # Style and label the subplot
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True)
        ax.set_ylabel("Frequency")
        # ax.set_xlabel(f"Position {start + 1}-{end}")
        # Set ticks at the first and last position of the chunk
        if show_axticks: 
            ax.set_xticks([0, chunk.shape[1] - 1])  # Positions relative to the chunk (0-based index)
            ax.set_xticklabels([start + 1, end])    # Labels corresponding to the actual sequence positions
        else: # Add blank to x ticks 
            ax.set_xticks([0])
            ax.set_xticklabels([''])
        ax.set_title(plot_title) if i == 0 else None # Only print label in the first plot 

    plt.tight_layout()

    return fig

def add_p_value_annotation(fig, 
                           array_columns, 
                           just_annotate = None,
                           test_type = 'ranksums', 
                           popmean = None, 
                           y_padding = True, 
                           subplot=None, 
                           include_tstat=None, 
                           p_round=3,
                           font_size=14, 
                           show=None, 
                           select_datatype=None,
                           horizontal=False,
                           _format=dict(interline=0.07, text_height=1.07, color='black')):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    horizontal: bool
        if True, annotate across y-axis (for horizontal plots); if False, annotate across x-axis (default)
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    
    assert test_type in ['ranksums', 'ttest_ind', 'ttest_rel', 'ttest_1samp'], \
        "Please specify test_type to be either ranksums or ttest"
    if test_type == 'ttest_1samp': 
        assert popmean is not None, "ttest_1samp requires popmean value"
    
    if just_annotate is not None: 
        assert len(just_annotate) == len(array_columns), "'just_annotate' and 'array_columns' len must be identical "


    # Filter data for subplots
    if subplot:
        subplot_str = '' if subplot == 1 else str(subplot)
        selected_data = [
            (i, data) for i, data in enumerate(fig.data) 
            if data['xaxis'] == 'x' + subplot_str
        ]
    else:
        subplot_str = ''
        selected_data = list(enumerate(fig.data))

    # Filter by datatype if specified
    if select_datatype:
        selected_data = [(i, d) for i, d in selected_data if d['type'] == select_datatype]
    
    # Extract only y-data and indices after filtering
    filtered_indices = [i for i, _ in selected_data]
    filtered_y_data = [d['y'] for _, d in selected_data]

    # Prepare annotation positions
    range_vals = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        base = 1.01 + (i * _format['interline'] if y_padding else _format['interline'])
        range_vals[i] = [base, base + 0.01]

    # Main loop for annotation
    for idx, column_pair in enumerate(array_columns):
        idx0, idx1 = column_pair
        y0 = filtered_y_data[idx0]
        y1 = filtered_y_data[idx1]
        
        if test_type == 'ttest_ind':
            tstat, pvalue = stats.ttest_ind(y0, y1, equal_var=False)
        elif test_type == 'ttest_rel':
            tstat, pvalue = stats.ttest_rel(y0, y1)
        elif test_type == 'ranksums':
            tstat, pvalue = stats.ranksums(y0, y1)
        elif test_type == 'ttest_1samp':
            tstat, pvalue = stats.ttest_1samp(y0, popmean=popmean)

        # Symbol formatting
        symbol = just_annotate[idx] if just_annotate else format_pvalue(pvalue, p_round, tstat if include_tstat else None, show)


        if horizontal:
            # Horizontal: annotate across y-axis
            if idx0 != idx1:
                for y in [idx0, idx1]:
                    fig.add_shape(type="line",
                        xref="x"+subplot_str+" domain", yref="y"+subplot_str,
                        x0=range_vals[idx][0]*_format['text_height'],
                        y0=y, 
                        x1=range_vals[idx][1]*_format['text_height'],
                        y1=y,
                        line=dict(color=_format['color'], width=2)
                    )
            fig.add_shape(type="line",
                xref="x"+subplot_str+" domain", yref="y"+subplot_str,
                x0=range_vals[idx][1]*_format['text_height'],
                y0=idx0,
                x1=range_vals[idx][1]*_format['text_height'],
                y1=idx1,
                line=dict(color=_format['color'], width=2)
            )
            fig.add_annotation(dict(
                font=dict(color=_format['color'], size=font_size),
                x=range_vals[idx][1]*(_format['text_height']+.07),
                y=(idx0 + idx1)/2,
                showarrow=False,
                text=symbol,
                textangle=0,
                xref="x"+subplot_str+" domain",
                yref="y"+subplot_str
            ))
        else:
            # Vertical: annotate across x-axis (original behavior)
            if idx0 != idx1:
                for x in [idx0, idx1]:
                    fig.add_shape(type="line",
                        xref="x"+subplot_str, yref="y"+subplot_str+" domain",
                        x0=x, y0=range_vals[idx][0]*_format['text_height'],
                        x1=x, y1=range_vals[idx][1]*_format['text_height'],
                        line=dict(color=_format['color'], width=2)
                    )
            fig.add_shape(type="line",
                xref="x"+subplot_str, yref="y"+subplot_str+" domain",
                x0=idx0, y0=range_vals[idx][1]*_format['text_height'],
                x1=idx1, y1=range_vals[idx][1]*_format['text_height'],
                line=dict(color=_format['color'], width=2)
            )
            fig.add_annotation(dict(
                font=dict(color=_format['color'], size=font_size),
                x=(idx0 + idx1)/2,
                y=range_vals[idx][1]*(_format['text_height']+.07),
                showarrow=False,
                text=symbol,
                textangle=0,
                xref="x"+subplot_str,
                yref="y"+subplot_str+" domain"
            ))
    return fig

def format_pvalue(pvalue, p_round=3, t=None, show=None, ns_to_p=False):
    """
    Format a p-value as a string with significance symbols.

    Parameters:
    pvalue (float): The p-value to be formatted.
    p_round (int): Number of digits to show in scientific notation.
    t (float, optional): Optional t-statistic to include.
    show (str or None): 
        If 'symbol', returns only the significance label (e.g. '*').
        If 'pvalue', returns only the formatted p-value (e.g. 'p=1.23e-03').
        If None, returns full string with significance and p-value (and t if provided).

    Returns:
    str: The formatted p-value string.
    """
    
    # Format p-value using scientific notation
    pval_part = f'p={pvalue:.1e}' if pvalue < 1e-2 else f'p={pvalue:.2f}'
    
    # Choose significance symbol
    if np.isnan(pvalue): 
        symbol_part= ''
    elif pvalue > 0.05:
        symbol_part = pval_part if ns_to_p else 'ns'
    elif pvalue > 0.01:
        symbol_part = '*'
    elif pvalue > 0.001:
        symbol_part = '**'
    else:
        symbol_part = '***'
    
    
    # Handle return based on return_part
    if show == 'symbol':
        return symbol_part
    elif show == 'pvalue':
        return pval_part
    
    # Default: full output
    symbol = f'{symbol_part} <br>{pval_part}'
    if t is not None:
        symbol += f'<br>t={round(t, 3)}'
    
    return symbol

from itertools import combinations
from scipy.spatial.distance import pdist

def compute_pairwise_pca_distance(pca_df, 
                                  zscore=True, colnames=['OR1', 'OR2', 'PCA_Distance']):
    """
    Compute pairwise Euclidean distances between PCA coordinates, with optional z-scoring.

    Parameters:
        pca_df (pd.DataFrame): DataFrame with PCA coordinates. Index should be entry names.
        zscore (bool): Whether to apply z-scoring to the distances. Default is True.
        colnames (list): List of 3 strings for naming the columns:
                         [name1, name2, distance_name]. Default is ['OR1', 'OR2', 'ESM_Distance'].

    Returns:
        pd.DataFrame: DataFrame with columns [name1, name2, distance_name].
    """
    if len(colnames) != 3:
        raise ValueError("colnames must be a list of 3 strings: [name1, name2, distance_name]")

    # Compute pairwise distances (returns condensed form)
    pca_dist = pdist(pca_df.values, metric='euclidean')

    # Optionally apply z-score
    if zscore:
        pca_dist = (pca_dist - np.mean(pca_dist)) / np.std(pca_dist)

    # Generate all unique index combinations
    names = pca_df.index.to_list()
    pairs = list(combinations(names, 2))

    # Assemble final DataFrame
    distance_df = pd.DataFrame(pairs, columns=colnames[:2])
    distance_df[colnames[2]] = pca_dist

    return distance_df