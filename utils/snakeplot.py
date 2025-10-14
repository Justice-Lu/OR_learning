import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import color_function as cf 

def smooth_curve(x0, x1, y0, y1, n, direction):
    m = min((n % 2 + 6), n)
    c = (x0 + x1) / 2
    r = 5
    circ_part_x = []
    circ_part_y = []
    for i in range(1, m + 1):
        circ_part_x.append(direction * np.cos(np.pi - i * np.pi / (m + 1)) * r + c)
        circ_part_y.append(direction * np.sin(np.pi - i * np.pi / (m + 1)) * r)
    if direction == -1:
        circ_part_x = circ_part_x[::-1]
        circ_part_y = circ_part_y[::-1]
    circ_part_x = np.array(circ_part_x)
    circ_part_y = np.array(circ_part_y)
    if n <= m:
        if direction == 1:
            return np.column_stack((circ_part_x, circ_part_y + max(y0, y1)))
        else:
            return np.column_stack((circ_part_x, circ_part_y + min(y0, y1)))
    ndiff = round((y1 - y0) / 4) * 2 * direction
    na = (n - m + ndiff) / 2
    nb = (n - m - ndiff) / 2
    if ndiff >= n - m:
        na = n - m
        nb = 0
    if ndiff < 0 and abs(ndiff) >= n - m:
        na = 0
        nb = n - m
    na = int(na)
    nb = int(nb)
    xa = np.full(na, x0)
    xb = np.full(nb, x1)
    if direction == 1:
        top = max(y0 + (na + 1) * 2, y1 + (nb + 1) * 2)
        circ_part_y = circ_part_y + top
        ya = np.linspace(y0 + 2, top, na)
        yb = np.linspace(top, y1 + 2, nb)
    else:
        bottom = min(y0 - (na - 1) * 2, y1 - (nb + 1) * 2)
        circ_part_y = circ_part_y + bottom
        ya = np.linspace(y0 - 2, bottom, na)
        yb = np.linspace(bottom, y1 - 2, nb)
    x_coords = np.concatenate((xa, circ_part_x, xb))
    y_coords = np.concatenate((ya, circ_part_y, yb))
    return np.column_stack((x_coords, y_coords))
def end_curve(x0, y0, n, direction):
    r = 5
    m = min((n % 2 + 6), n)
    xa = np.full(n - m, x0)
    ya = np.array([y0 + i * direction * 2 for i in range(n - m)])
    circ_part_x = []
    circ_part_y = []
    for i in range(1, m + 1):
        circ_part_x.append(direction * np.cos(np.pi - i * np.pi / (m + 1)) * r + x0 - direction * r)
        circ_part_y.append(direction * np.sin(np.pi - i * np.pi / (m + 1)) * r)
    circ_part_x = np.array(circ_part_x)
    circ_part_y = np.array(circ_part_y)
    if direction == -1:
        circ_part_y = circ_part_y + min(np.concatenate((ya, [y0])))
        x_coords = np.concatenate((xa, circ_part_x[::-1]))
        y_coords = np.concatenate((ya, circ_part_y[::-1]))
    else:
        circ_part_y = circ_part_y + max(np.concatenate((ya, [y0])))
        x_coords = np.concatenate((circ_part_x, xa[::-1]))
        y_coords = np.concatenate((circ_part_y, ya[::-1]))
    return np.column_stack((x_coords, y_coords))


def tm_part(x0, y0, n, direction):
    pattern_x = np.array([2, 0, -2, 3, 1, -1, -3])
    pattern_y = np.array([0, -1.2, -2.4, -1.8, -3.0, -4.2, -5.4])
    tmx = np.tile(pattern_x, int(np.ceil(n / 7)))[:n] + x0
    tmy = np.tile(pattern_y, int(np.ceil(n / 7)))[:n]
    for i in range(int(np.ceil(n / 7))):
        start = 7 * i
        end = min(n, 7 * (i + 1))
        tmy[start:end] -= i * 5.4
    tmy += y0
    if direction == -1:
        tmx = tmx[::-1]
        tmy = tmy[::-1]
    return np.column_stack((tmx, tmy))

def tm_to_center(TM, y_center):
    diff = np.mean(TM[:, 1]) - y_center
    TM[:, 1] = TM[:, 1] - diff
    return TM

def snake_coords(ec1, tm1, ic1, tm2, ec2, tm3,
               ic2, tm4, ec3, tm5, ic3, tm6,
               ec4, tm7, ic4):
    # build TM segments
    TM1 = tm_part(20, 90, tm1, 1)
    TM2 = tm_part(30, 90, tm2, -1)
    TM3 = tm_part(40, 90, tm3, 1)
    TM4 = tm_part(50, 90, tm4, -1)
    TM5 = tm_part(60, 90, tm5, 1)
    TM6 = tm_part(70, 90, tm6, -1)
    TM7 = tm_part(80, 90, tm7, 1)

    # vertical centering
    y_center = np.mean(np.concatenate([TM1[:, 1], TM2[:, 1], TM3[:, 1],
                                       TM4[:, 1], TM5[:, 1], TM6[:, 1], TM7[:, 1]]))
    TM1 = tm_to_center(TM1, y_center)
    TM2 = tm_to_center(TM2, y_center)
    TM3 = tm_to_center(TM3, y_center)
    TM4 = tm_to_center(TM4, y_center)
    TM5 = tm_to_center(TM5, y_center)
    TM6 = tm_to_center(TM6, y_center)
    TM7 = tm_to_center(TM7, y_center)

    # loop segments
    EC1 = end_curve(20, TM1[0, 1] + 1, ec1, 1)
    IC1 = smooth_curve(20, 30, TM1[-1, 1] - 1, TM2[0, 1] - 1, ic1, -1)
    EC2 = smooth_curve(30, 40, TM2[-1, 1], TM3[0, 1], ec2, 1)
    IC2 = smooth_curve(40, 50, TM3[-1, 1] - 1, TM4[0, 1] - 1, ic2, -1)
    EC3 = smooth_curve(50, 60, TM4[-1, 1], TM5[0, 1], ec3, 1)
    IC3 = smooth_curve(60, 70, TM5[-1, 1] - 1, TM6[0, 1] - 1, ic3, -1)
    EC4 = smooth_curve(70, 80, TM6[-1, 1], TM7[0, 1], ec4, 1)
    IC4 = end_curve(80, TM7[-1, 1] - 2, ic4, -1)

    # assemble and track TM starts
    segments = [
        ("EC1", EC1), ("TM1", TM1), ("IC1", IC1),
        ("TM2", TM2), ("EC2", EC2), ("TM3", TM3),
        ("IC2", IC2), ("TM4", TM4), ("EC3", EC3),
        ("TM5", TM5), ("IC3", IC3), ("TM6", TM6),
        ("EC4", EC4), ("TM7", TM7), ("IC4", IC4)
    ]

    coords = []
    tm_starts = []
    idx = 0
    for name, seg in segments:
        if name.startswith("TM"):  # mark each TM start
            tm_starts.append(idx)
        coords.append(seg)
        idx += len(seg)
    coords = np.vstack(coords)

    return coords

def make_snakedf(segments=[23, 28, 6, 29, 8, 34, 10, 22, 36, 30, 7, 33, 11, 19, 21], 
                 snake_labels=['M', 'E', 'M', 'G', 'N', 'Q', 'T', 'S', 'V', 'T', 'E', 'F', 'I', 'L', 'L', 'G', 'L', 'S', 'D', 'D', 'P', 'E', 'L', 'Q', 'L', 'L', 'L', 'F', 'V', 'L', 'F', 'L', 'L', 'I', 'Y', 'L', 'V', 'T', 'L', 'L', 'G', 'N', 'L', 'L', 'I', 'I', 'L', 'L', 'I', 'T', 'L', 'D', 'S', 'H', 'L', 'H', 'T', 'P', 'M', 'Y', 'F', 'F', 'L', 'S', 'N', 'L', 'S', 'F', 'L', 'D', 'I', 'C', 'Y', 'S', 'S', 'V', 'T', 'V', 'P', 'K', 'M', 'L', 'V', 'N', 'F', 'L', 'S', 'E', 'K', 'K', 'K', 'T', 'I', 'S', 'F', 'A', 'G', 'C', 'M', 'T', 'Q', 'L', 'F', 'F', 'F', 'H', 'F', 'F', 'G', 'G', 'T', 'E', 'C', 'F', 'L', 'L', 'A', 'A', 'M', 'A', 'Y', 'D', 'R', 'Y', 'V', 'A', 'I', 'C', 'K', 'P', 'L', 'H', 'Y', 'T', 'T', 'I', 'M', 'S', 'P', 'R', 'V', 'C', 'V', 'L', 'L', 'V', 'L', 'G', 'S', 'W', 'V', 'G', 'G', 'F', 'L', 'L', 'A', 'S', 'L', 'I', 'H', 'T', 'L', 'L', 'T', 'L', 'R', 'L', 'P', 'F', 'C', 'G', 'S', 'N', 'V', 'I', 'N', 'H', 'F', 'F', 'C', 'D', 'I', 'P', 'P', 'L', 'L', 'K', 'L', 'A', 'C', 'S', 'D', 'T', 'S', 'I', 'N', 'E', 'L', 'V', 'V', 'F', 'V', 'V', 'A', 'G', 'F', 'I', 'L', 'L', 'V', 'P', 'F', 'L', 'L', 'I', 'L', 'V', 'S', 'Y', 'I', 'F', 'I', 'L', 'S', 'T', 'I', 'L', 'R', 'I', 'P', 'S', 'A', 'E', 'G', 'R', 'R', 'K', 'A', 'F', 'S', 'T', 'C', 'S', 'S', 'H', 'L', 'T', 'V', 'V', 'S', 'L', 'F', 'Y', 'G', 'T', 'A', 'I', 'F', 'M', 'Y', 'L', 'Y', 'P', 'P', 'P', 'S', 'S', 'S', 'S', 'S', 'D', 'Q', 'D', 'K', 'V', 'V', 'S', 'V', 'F', 'Y', 'T', 'V', 'V', 'T', 'P', 'M', 'L', 'N', 'P', 'L', 'I', 'Y', 'S', 'L', 'R', 'N', 'K', 'D', 'V', 'K', 'G', 'A', 'L', 'K', 'K', 'L', 'L', 'G', 'R', 'K', 'K', 'S', 'S', 'S', 'K', 'K']):
    """
    Generate the full snakeplot DataFrame.

    Parameters
    ----------
    segments : list of int
        Segment lengths for [EC1, TM1, IC1, TM2, ..., TM7, IC4].
    snake_labels : str, default=""
        Sequence to annotate positions with.

    Returns
    -------
    snakeplot_df : pd.DataFrame
        DataFrame with x, y, segment, text, and position.
    """
    assert len(segments) == 15, \
        "Segments must be len 15: [EC1, TM1, IC1, TM2, EC2, TM3, IC2, TM4, EC3, TM5, IC3, TM6, EC4, TM7, IC4]"

    # raw coords from curves
    coords = snake_coords(*segments)  # helper = your original snake_coords internals
    snakeplot_df = pd.DataFrame(coords, columns=["x", "y"])

    # segment labels
    segment_labels = ["EC1", "TM1", "IC1", "TM2", "EC2", "TM3",
                      "IC2", "TM4", "EC3", "TM5", "IC3", "TM6",
                      "EC4", "TM7", "IC4"]

    snakeplot_df["segments"] = np.concatenate([
        [segment_labels[i]] * seg_len for i, seg_len in enumerate(segments)
    ])

    # sequence text
    seq = list(snake_labels)
    snakeplot_df["text"] = seq + [""] * (len(snakeplot_df) - len(seq)) if len(seq) <= len(snakeplot_df) else seq
    snakeplot_df["position"] = range(1, len(snakeplot_df) + 1)

    return snakeplot_df

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

def plt_snakeplot(df,
                  ax=None, 
                  label_position=[], 
                  label_color="Red", 
                  figsize=(12,10),
                  point_size=150,
                  point_alpha=1.0,
                  line_color="#D3D3D3",
                  line_width=2,
                  color_by=None,
                  manual_color=None,
                  cmap='tab20',
                  cmap_midpoint=None, 
                  cmin=None, cmax=None, 
                  edgecolors=None,
                  fontweight='normal',
                  text_size=10,
                  show_axis=False,
                  title=None,
                  title_text_size=20,
                  show_colorbar=False):
    """
    Snakeplot with continuous color scaling and proper NaN handling.
    """

    # assign colors
    if color_by and manual_color is None:
        col_data = df[color_by]
        if np.issubdtype(col_data.dtype, np.number):
            # continuous numeric
            df["color"] = cf.get_continuous_colors(
                col_data, 
                colormap=cmap,
                midpoint=cmap_midpoint,
                color_min=cmin,
                color_max=cmax
            )
            df["color"] = df["color"].fillna("#D3D3D3")  # default for NaN
            is_continuous = True
        else:
            # categorical
            colormap_dict = cf.distinct_colors(col_data.dropna().unique(), category=cmap)
            df["color"] = col_data.map(colormap_dict)
            df["color"] = df["color"].fillna("#D3D3D3")
            is_continuous = False
    elif manual_color is not None:
        df["color"] = manual_color
        is_continuous = False
    else:
        df["color"] = "skyblue"
        is_continuous = False

    # create axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # plot backbone line
    ax.plot(df['x'], df['y'], '-', color=line_color, linewidth=line_width, zorder=1)

    # scatter points
    sc = ax.scatter(df['x'], df['y'], s=point_size, alpha=point_alpha,
                    c=df['color'], edgecolors=edgecolors, zorder=2)

    # highlight positions
    # highlight positions with concentric rings if multiple labels overlap
    highlight_counts = {}  # track how many times a position has been labeled
    
    if isinstance(label_position[0], (list, tuple, np.ndarray)):
        for pos_group, col in zip(label_position, label_color):
            for _pos in pos_group:
                subset_df = df[df['position'] == _pos]
                if subset_df.empty:
                    continue
                # increment count
                count = highlight_counts.get(_pos, 0) + 1
                highlight_counts[_pos] = count

                # expand circle size each time it's added
                ax.scatter(subset_df['x'], subset_df['y'], 
                           s=point_size*(1.5 + 1*(count-1)), alpha=1,
                           linewidths=2, facecolor='None', 
                           edgecolors=col, zorder=3)
    else:
        # single category (backward compatibility)
        for _pos in label_position:
            subset_df = df[df['position'] == _pos]
            ax.scatter(subset_df['x'], subset_df['y'], 
                       s=point_size*1.5, alpha=1,
                       linewidths=3, facecolor='None', 
                       edgecolors=label_color, zorder=3)

    # text labels
    for _, row in df.iterrows():
        if row['text'] and row['text'] != ' ':
            ax.text(row['x'], row['y'], row['text'],
                    color='white' if cf.hex_contrast(row['color']) == 'dark' else 'black',
                    fontweight=fontweight, fontsize=text_size,
                    ha='center', va='center', zorder=3)

    if not show_axis:
        ax.axis('off')
    if title:
        ax.set_title(title, fontdict=dict(fontsize=title_text_size))

    # colorbar
    if show_colorbar and is_continuous:
        vals = df[color_by].dropna()
        vmin = cmin if cmin is not None else vals.min()
        vmax = cmax if cmax is not None else vals.max()
        
        midpoint = (vmin+vmax)//2
        if cmap_midpoint is not None: 
            if cmap_midpoint > vmin: 
                midpoint = cmap_midpoint
            
        # midpoint = cmap_midpoint if (cmap_midpoint is not None) & (cmap_midpoint > vmin) else (vmin+vmax)//2

        # Use diverging normalization if midpoint is within vmin/vmax
        if vmin < midpoint < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', 
                            panchor=(0,1),aspect=4, 
                            fraction=0.04, pad=0.01)
        # auto-tick: min, midpoint (if in range), max
        ticks = [vmin]
        if vmin < midpoint < vmax:
            ticks.append(midpoint)
        ticks.append(vmax)
        # convert to int for aesthetics 
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
        cbar.set_label(color_by)

    return ax

import pandas as pd
from collections import defaultdict

def snake_df_label_bw(snake_df: pd.DataFrame, bw_anchor_dict: dict, label = 'bw') -> pd.DataFrame:
    """
    Populate a single 'bw' column in snakeplot dataframe, only within the segment
    of each BW anchor. Each step = ±0.01.
    
    Parameters
    ----------
    snake_df : pd.DataFrame
        Must have columns 'position' and 'segments'.
    bw_anchor_dict : dict
        Mapping of anchor BW (string, e.g. '1.50') -> residue position (int)
    """
    df = snake_df.copy()
    bw_col = {}

    # Organize anchors by segment
    segment_anchors = defaultdict(list)
    for bw_str, pos in bw_anchor_dict.items():
        pos_int = int(pos)
        subset = df[df['position'] == pos_int]
        if subset.empty:
            continue
        seg = subset['segments'].iloc[0]
        segment_anchors[seg].append((pos_int, float(bw_str)))

    # Process each segment independently
    for seg, anchors in segment_anchors.items():
        anchors_sorted = sorted(anchors, key=lambda x: x[0])
        seg_df = df[df['segments'] == seg].sort_values('position')
        positions = seg_df['position'].values
        pos_to_idx = {int(p): i for i, p in enumerate(positions)}

        # Assign anchors
        for pos, bw_val in anchors_sorted:
            bw_col[pos] = bw_val

        # Fill backward from first anchor
        first_pos, first_bw = anchors_sorted[0]
        first_idx = pos_to_idx[first_pos]
        for idx in range(first_idx - 1, -1, -1):
            bw_val = bw_col[positions[idx + 1]] - 0.01
            bw_col[positions[idx]] = bw_val

        # Fill forward from last anchor
        last_pos, last_bw = anchors_sorted[-1]
        last_idx = pos_to_idx[last_pos]
        for idx in range(last_idx + 1, len(positions)):
            bw_val = bw_col[positions[idx - 1]] + 0.01
            bw_col[positions[idx]] = bw_val

        # Fill between multiple anchors
        if len(anchors_sorted) >= 2:
            for k in range(len(anchors_sorted) - 1):
                pos_i, bw_i = anchors_sorted[k]
                pos_j, bw_j = anchors_sorted[k + 1]
                idx_i = pos_to_idx[pos_i]
                idx_j = pos_to_idx[pos_j]
                for t in range(idx_i + 1, idx_j):
                    bw_col[positions[t]] = bw_col[positions[t - 1]] + 0.01

    # Map final bw values as formatted string
    df[label] = df['position'].map(lambda x: f"{bw_col[x]:.2f}" if x in bw_col else None)
    return df