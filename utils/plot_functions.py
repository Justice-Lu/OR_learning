import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

import logomaker

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