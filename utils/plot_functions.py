import numpy as np 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


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


def visualize_voxel_grid(voxel_data, 
                         coordinate_labels, 
                         color_map, 
                         opacity=0.1,
                         highlight_labels=None, 
                         highlight_opacity=0.5):
    """
    Visualizes 3D voxel grids using Plotly Scatter3D.

    :param voxel_data: 
        List of voxel grids. Each grid is a numpy array of shape (X, Y, Z).
    :param coordinate_labels: 
        Dictionary where keys are labels (e.g., "BACKBONE", "BOUND_POINT") and values are voxel grids.
    :param color_map: 
        Dictionary mapping labels to colors for visualization.
    :param highlight_labels: 
        List of labels to highlight with higher opacity. Default is None.
    """
    fig = go.Figure()

    for i, (label, voxel_grid) in enumerate(zip(coordinate_labels, voxel_data)):
        # Get occupied voxels
        occupied_voxels = np.array(np.where(voxel_grid == 1)).T
        
        if len(occupied_voxels) == 0:
            continue  # Skip empty grids
        
        # Extract x, y, z coordinates
        x = occupied_voxels[:, 0]
        y = occupied_voxels[:, 1]
        z = occupied_voxels[:, 2]
        
        # Add scatter plot for the current label
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=label,
            marker=dict(
                size=5,
                color=color_map.get(label, 'gray'),  # Default color is gray if label is not in the color_map
                opacity= highlight_opacity if highlight_labels and label in highlight_labels else opacity
            )
        ))
    
    # Apply a blank style 
    fig = _plotly_blank_style(fig)
    
    return fig

def plot_correlation(values1, values2,
                     xlabel='', 
                     ylabel='',
                     title='', 
                     plot_pearson_line=True, 
                     edgecolor='gray',
                     edgesize=10, 
                     opacity=0.5, 
                     text_xy=[0.05, 0.95], 
                     figsize=[8,8]):
    """
    Creates a scatter plot with a line of best fit and Pearson correlation annotation.

    :param values1: Array-like, the first set of values (e.g., Grantham distances).
    :param values2: Array-like, the second set of values (e.g., response correlations).
    :param xlabel: String, label for the x-axis.
    :param ylabel: String, label for the y-axis.
    :param title: String, title of the plot.
    :return: None, displays the plot.
    """
    # Ensure inputs are numpy arrays for consistency
    values1 = np.array(values1)
    values2 = np.array(values2)

    # Calculate Pearson correlation and p-value
    r, p_value = pearsonr(values1, values2)

    # Calculate the line of best fit
    slope, intercept, _, _, _ = linregress(values1, values2)

    # Create the scatter plot
    plt.figure(figsize=(figsize[0], figsize[1]))
    plt.scatter(values1, values2, color=edgecolor, alpha=opacity, s=edgesize)
    
    # Add labels, title, and grid
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # Add text annotation for Pearson correlation, r^2, and p-value
    if plot_pearson_line: 
        plt.plot(values1, slope * values1 + intercept, color="black", linestyle="--", label="Pearson Line", alpha=0.6)
        plt.text(
            text_xy[0], text_xy[1],  # Adjust position as needed
            f"Pearson r = {r:.3f}\n$r^2$ = {r**2:.3f}\np-value = {p_value:.3e}",
            fontsize=12,
            ha="left", va="top",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.5),
        )

    # Add tight layout
    plt.tight_layout()

    # Show the plot
    return plt