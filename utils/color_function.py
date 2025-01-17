import plotly.colors
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
        

# Identical to Adam's answer

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )

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

    Example:
    >>> labels = ['A', 'B', 'C']
    >>> color_mapping = distinct_colors(labels, category='tab10')
    >>> print(color_mapping)
    {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
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

    color_palette = None

    # Handle custom categories
    if category in ['warm', 'floral', 'rainbow', 'pastel']: 
        if category == 'warm':
            color_palette = random.sample(warm_colors, len(warm_colors))
        elif category == 'floral':
            color_palette = random.sample(floral_colors, len(floral_colors))
        elif category == 'rainbow':
            color_palette = random.sample(rainbow_colors, len(rainbow_colors))
        elif category == 'pastel': 
            color_palette = random.sample(pastel_colors, len(pastel_colors))
        # else:
        #     color_palette = random.sample(warm_colors + floral_colors + rainbow_colors + pastel_colors, len(label_list))
    # Handle matplotlib color palettes
    elif category in mcolors.TABLEAU_COLORS or category in cm.cmaps_listed or hasattr(cm, category):
        cmap = cm.get_cmap(category) if hasattr(cm, category) else cm.get_cmap('tab10')
        num_colors = len(label_list)
        color_palette = [mcolors.to_hex(cmap(i / num_colors)) for i in range(num_colors)]
    
    for i, label in enumerate(label_list):
        color_dict[label] = color_palette[i % len(color_palette)]
    
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
