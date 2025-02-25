import numpy as np 
import plotly.colors as pc 
import random
from matplotlib import cm, colors as mcolors

from PIL import ImageColor


def get_color(colorscale_name, loc):
    """
# This function allows you to retrieve colors from a continuous color scale
# by providing the name of the color scale, and the normalized location between 0 and 1
# Reference: https://stackoverflow.com/questions/62710057/access-color-from-plotly-color-scale
    """


    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)
        


def get_continuous_colors(values, colormap="RdBu_r", midpoint=None):
    """
    Assigns colors to a list or dictionary of values using a continuous colorscale.

    Parameters:
    - values (list, dict, or Series): A dictionary {index: value}, Pandas Series, or a list of values.
    - colormap (str): A Matplotlib colormap name (e.g., 'RdBu_r', 'viridis', 'coolwarm').
    - midpoint (float, optional): The value to center the colormap around. If None, uses the median.

    Returns:
    - dict: A dictionary mapping indices (or positions if input is a list) to colors in hex format.
    """
    # Convert Pandas Series to dictionary if applicable
    if hasattr(values, "to_dict"):
        values = values.to_dict()

    # Extract indices and values
    if isinstance(values, dict):
        indices, vals = list(values.keys()), np.array(list(values.values()))
    else:
        indices, vals = range(len(values)), np.array(values)

    # Handle edge case where all values are the same
    if np.all(vals == vals[0]):
        return {idx: mcolors.to_hex(cm.get_cmap(colormap)(0.5)) for idx in indices}

    # Define midpoint if not provided (default: median)
    if midpoint is None:
        midpoint = np.median(vals)

    # Determine min and max values
    min_val, max_val = np.min(vals), np.max(vals)

    # Adjust the color scale if all values are on one side of the midpoint
    if min_val >= midpoint:  # All values are positive
        max_val = max(max_val, abs(midpoint))  # Ensure symmetry
        min_val = -max_val
    elif max_val <= midpoint:  # All values are negative
        min_val = min(min_val, -abs(midpoint))  # Ensure symmetry
        max_val = -min_val

    # Normalize values between [0, 1] while keeping midpoint centered
    if min_val < midpoint < max_val:
        norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=midpoint, vmax=max_val)
    else:
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # Get the colormap and apply normalization
    cmap = cm.get_cmap(colormap)
    colors = {idx: mcolors.to_hex(cmap(norm(value))) for idx, value in zip(indices, vals)}

    return colors

def distinct_colors(label_list, category='tab10', custom_color=None, random_state=0):
    """
    Generate distinct colors for a list of labels.

    Parameters:
    label_list (list): A list of labels for which you want to generate distinct colors.
    category (str): Category of distinct colors. Options are 'warm', 'floral', 'rainbow', 'pastel',
                    matplotlib color palettes (e.g., 'tab10', 'Set2'), or None for random. Default is None.
    custom_color (list): A custom list of colors to use.
    random_state (int): Seed for random color generation. Default is 0.

    Returns:
    dict: A dictionary where labels are keys and distinct colors (in hexadecimal format) are values.
    """
    random.seed(random_state)
    
    warm_colors = ['#fabebe', '#ffd8b1', '#fffac8', '#ffe119', '#ff7f00', '#e6194B']
    floral_colors = ['#bfef45', '#fabed4', '#aaffc3', '#ffd8b1', '#dcbeff', '#a9a9a9']
    rainbow_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
    pastel_colors = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', 
                     '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928', 
                     '#8DD3C7', '#BEBADA', '#FFED6F']
    
    color_dict = {}

    if custom_color is not None: 
        assert len(custom_color) >= len(label_list), "Provided label_list needs to be shorter than provided custom_color"
        for i, _label in enumerate(label_list): 
            color_dict[_label] = custom_color[i]
        return color_dict

    color_palette = []

    # Handle predefined categories
    if category in ['warm', 'floral', 'rainbow', 'pastel']: 
        if category == 'warm':
            color_palette = warm_colors
        elif category == 'floral':
            color_palette = floral_colors
        elif category == 'rainbow':
            color_palette = rainbow_colors
        elif category == 'pastel': 
            color_palette = pastel_colors

        # If more labels than available colors, interpolate colors
        if len(label_list) > len(color_palette):
            cmap = cm.get_cmap("tab20")  # Use a larger colormap
            num_colors = len(label_list)
            color_palette = [mcolors.to_hex(cmap(i / num_colors)) for i in range(num_colors)]
    
    # Handle matplotlib colormaps
    elif category in cm.cmaps_listed or hasattr(cm, category):
        cmap = cm.get_cmap(category) if hasattr(cm, category) else cm.get_cmap('tab10')
        num_colors = len(label_list)
        
        # Ensure unique colors using interpolation
        color_palette = [mcolors.to_hex(cmap(i / (num_colors - 1))) for i in range(num_colors)]

    # Assign distinct colors to each label
    for i, label in enumerate(label_list):
        color_dict[label] = color_palette[i]
    
    return color_dict

def scale(values, reverse=False, factor = 1, scale_between = [1,0]):
    """
    Reverses the scale of a list of values such that the smallest value becomes 1 and the largest value becomes 0.
    """
    
    min_val = min(values)
    max_val = max(values)
    new_max = scale_between[0]
    new_min = scale_between[1]

    if min_val != max_val: #if all the min and max is the same value. assign  max size 
        scaled_values = [(value - min_val) * (new_max - new_min) / (max_val - min_val) + new_min for value in values]
    else: 
        scaled_values = [value*new_max for value in values]
#     scaled_values = [(val - min_val) / (max_val - min_val)*factor for val in values]
    if reverse:
        scaled_values = [1 - val for val in scaled_values]
    return scaled_values
