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
        


from matplotlib import cm, colors as mcolors
import numpy as np
import pandas as pd

def get_continuous_colors(values, colormap="RdBu_r", midpoint=None, color_min=None, color_max=None):
    """
    Assign colors to values using a continuous colormap, handling NaNs and skewed data.

    Parameters:
    - values: list, np.array, or pd.Series of numeric values (NaNs allowed)
    - colormap: matplotlib colormap string
    - midpoint: float, center of the colormap (e.g., 0 for diverging maps)
    - color_min: float, optional user-defined minimum value for colormap
    - color_max: float, optional user-defined maximum value for colormap

    Returns:
    - dict: mapping index -> hex color
    """
    # Convert to array
    if isinstance(values, (pd.Series, list)):
        indices = np.arange(len(values))
        vals = np.array(values, dtype=float)
    elif isinstance(values, np.ndarray):
        indices = np.arange(len(values))
        vals = values.astype(float)
    else:
        raise ValueError("Unsupported input type for values.")

    # Handle NaNs
    mask = ~np.isnan(vals)
    vals_masked = vals[mask]

    # Determine colormap bounds
    vmin = color_min if color_min is not None else np.min(vals_masked)
    vmax = color_max if color_max is not None else np.max(vals_masked)
    
    # Determine midpoint
    if midpoint is None:
        midpoint = 0  # default center for diverging importance

    # Use TwoSlopeNorm if midpoint is within vmin/vmax, else simple Normalize
    if vmin < midpoint < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colors = np.full(vals.shape, "#D3D3D3", dtype=object)  # default black for NaN
    colors[mask] = [mcolors.to_hex(cmap(norm(v))) for v in vals_masked]

    return {idx: color for idx, color in zip(indices, colors)}

def distinct_colors(label_list=None, category='tab10', custom_color=None, random_state=0, form='dict', num_colors=None):
    """
    Generate distinct colors for a list of labels or a specified number of colors.

    Parameters:
    label_list (list, optional): A list of labels for which you want to generate distinct colors.
                                If None, will generate num_colors distinct colors.
    category (str): Category of distinct colors. Options are 'warm', 'floral', 'rainbow', 'pastel',
                    matplotlib color palettes (e.g., 'tab10', 'Set2'), or 'random'. Default is 'tab10'.
    custom_color (list, optional): A custom list of colors to use.
    random_state (int): Seed for random color generation. Default is 0.
    form (str): Output format - 'dict' for a label-to-color dictionary, 'list' for a list of colors. Default is 'dict'.
    num_colors (int, optional): Number of colors to generate if label_list is None.

    Returns:
    dict or list: Colors in the requested format (dictionary mapping labels to colors or list of colors).
    """
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Predefined color palettes
    warm_colors = ['#fabebe', '#ffd8b1', '#fffac8', '#ffe119', '#ff7f00', '#e6194B']
    floral_colors = ['#bfef45', '#fabed4', '#aaffc3', '#ffd8b1', '#dcbeff', '#a9a9a9']
    rainbow_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
    pastel_colors = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', 
                     '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928', 
                     '#8DD3C7', '#BEBADA', '#FFED6F']
    
    # Determine the number of colors needed
    if label_list is not None:
        required_colors = len(label_list)
    elif num_colors is not None:
        required_colors = num_colors
    else:
        raise ValueError("Either label_list or num_colors must be provided")
    
    # Handle custom colors
    if custom_color is not None:
        if len(custom_color) < required_colors:
            raise ValueError(f"Not enough custom colors ({len(custom_color)}) for the required number ({required_colors})")
        color_palette = custom_color[:required_colors]
    
    # Generate color palette based on category
    else:
        if category == 'warm':
            base_palette = warm_colors
        elif category == 'floral':
            base_palette = floral_colors
        elif category == 'rainbow':
            base_palette = rainbow_colors
        elif category == 'pastel':
            base_palette = pastel_colors
        elif category == 'random':
            # Generate completely random colors
            color_palette = ["#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(required_colors)]
        else:
            # Try to use matplotlib colormap
            try:
                cmap = cm.get_cmap(category)
                base_palette = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, min(required_colors, 20))]
            except (ValueError, AttributeError):
                # Fallback to tab10 if the specified category is not available
                cmap = cm.get_cmap('tab10')
                base_palette = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, min(required_colors, 10))]
        
        # If category wasn't 'random', ensure we have enough unique colors
        if category != 'random':
            color_palette = []
            
            # If we need more colors than in the base palette, use interpolation and HSV manipulation
            if required_colors > len(base_palette):
                # Start with all colors from the base palette
                color_palette = base_palette.copy()
                
                # Convert to HSV for better interpolation and manipulation
                hsv_colors = [mcolors.rgb_to_hsv(mcolors.to_rgb(color)) for color in base_palette]
                
                # Generate additional colors by manipulating hue and saturation
                while len(color_palette) < required_colors:
                    new_hsv = hsv_colors[len(color_palette) % len(hsv_colors)].copy()
                    # Modify hue and saturation slightly
                    new_hsv[0] = (new_hsv[0] + 0.1 * (len(color_palette) // len(hsv_colors))) % 1.0
                    new_hsv[1] = max(0.4, min(1.0, new_hsv[1] + 0.05 * ((len(color_palette) // len(hsv_colors)) % 3 - 1)))
                    
                    # Convert back to RGB, then hex
                    new_rgb = mcolors.hsv_to_rgb(new_hsv)
                    new_hex = mcolors.to_hex(new_rgb)
                    
                    # Only add if the color is visually distinct enough (simple check)
                    if all(mcolors.rgb_to_hsv(mcolors.to_rgb(new_hex))[0] != 
                           mcolors.rgb_to_hsv(mcolors.to_rgb(existing))[0] for existing in color_palette[-10:]):
                        color_palette.append(new_hex)
                    else:
                        # If too similar, add some randomness to the hue
                        new_hsv[0] = (new_hsv[0] + random.random() * 0.2) % 1.0
                        new_rgb = mcolors.hsv_to_rgb(new_hsv)
                        color_palette.append(mcolors.to_hex(new_rgb))
            else:
                # If we have enough colors in the base palette, just use those
                color_palette = base_palette[:required_colors]
    
    # Return results in the requested format
    if label_list is None:
        return color_palette if form == 'list' else {i: color for i, color in enumerate(color_palette)}
    else:
        if form == 'list':
            return color_palette
        else:
            return {label: color_palette[i] for i, label in enumerate(label_list)}
        
import colorsys

def generate_faded_shades(hex_color, shades, lightness_range=(0.3, 0.9)):
    """
    Generate lighter (faded) hex color shades from a base color.

    Parameters:
    - hex_color: base hex color string (e.g. '#1f77b4')
    - shades: int for number of shades, or list of strings for labeled shades
    - lightness_range: (min, max) lightness to span the faded shades

    Returns:
    - List of hex colors if shades is int
    - Dict {label: hex} if shades is list of strings
    """
    if isinstance(shades, int):
        n_shades = shades
        keys = None
    elif isinstance(shades, list) and all(isinstance(k, str) for k in shades):
        n_shades = len(shades)
        keys = shades
    else:
        raise ValueError("`shades` must be either an integer or a list of strings.")

    # Convert hex to RGB
    hex_color_clean = hex_color.lstrip('#')
    r, g, b = [int(hex_color_clean[i:i+2], 16)/255. for i in (0, 2 ,4)]

    # Convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Compute linearly spaced lightness values
    lightness_values = [
        lightness_range[0] + i * (lightness_range[1] - lightness_range[0]) / max(1, n_shades - 1)
        for i in range(n_shades)
    ]

    # Generate colors
    faded_colors = []
    for lv in lightness_values:
        # You may want to reduce saturation slightly to keep it from looking neon
        r_f, g_f, b_f = colorsys.hls_to_rgb(h, lv, s * 0.9)
        hex_faded = '#{0:02x}{1:02x}{2:02x}'.format(int(r_f*255), int(g_f*255), int(b_f*255))
        faded_colors.append(hex_faded)

    return dict(zip(keys, faded_colors)) if keys else faded_colors


# from plotly.validators.scatter.marker import SymbolValidator
from plotly.validator_cache import ValidatorCache

import random

def distinct_shapes(label_list=None, random_state=0, form='dict', num_shapes=None):
    """
    Generate distinct shapes for a list of labels or a specified number of shapes.

    Parameters:
    label_list (list, optional): A list of labels for which you want to generate distinct shapes.
                                 If None, will generate num_shapes distinct shapes.
    random_state (int): Seed for reproducibility. Default is 0.
    form (str): Output format - 'dict' for a label-to-shape dictionary, 'list' for a list of shapes. Default is 'dict'.
    num_shapes (int, optional): Number of shapes to generate if label_list is None.

    Returns:
    dict or list: Shapes in the requested format (dictionary mapping labels to shapes or a list of shapes).
    """
    
    random.seed(random_state)

    # Get all available marker symbols in Plotly
    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
    all_shapes = [s for i, s in enumerate(SymbolValidator().values) if i % 3 == 2]  
    
    # Separate main shapes (without '-') and sub-shapes (with '-')
    main_shapes = [s for s in all_shapes if '-' not in s]
    sub_shapes = [s for s in all_shapes if '-' in s]

    # Determine number of required shapes
    if label_list is not None:
        required_shapes = len(label_list)
    elif num_shapes is not None:
        required_shapes = num_shapes
    else:
        raise ValueError("Either label_list or num_shapes must be provided.")

    # Prioritize main shapes, then use sub-shapes if needed
    shape_list = main_shapes[:required_shapes]  # Take as many main shapes as available
    if len(shape_list) < required_shapes:
        remaining = required_shapes - len(shape_list)
        shape_list.extend(sub_shapes[:remaining])  # Fill remaining slots with sub-shapes

    # Ensure cycling if still not enough
    if len(shape_list) < required_shapes:
        combined_shapes = main_shapes + sub_shapes
        shape_list = [combined_shapes[i % len(combined_shapes)] for i in range(required_shapes)]

    # Return the requested format
    if label_list is None:
        return shape_list if form == 'list' else {i: shape for i, shape in enumerate(shape_list)}
    else:
        return {label: shape_list[i] for i, label in enumerate(label_list)} if form == 'dict' else shape_list

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

def hex_contrast(hex_color: str) -> str:
    """Return 'dark' or 'bright' for a given hex color."""
    rgb = mcolors.hex2color(hex_color)  # values in [0,1]
    r, g, b = [int(c*255) for c in rgb]

    # luminance formula
    luminance = 0.299*r + 0.587*g + 0.114*b
    return "bright" if luminance > 186 else "dark"