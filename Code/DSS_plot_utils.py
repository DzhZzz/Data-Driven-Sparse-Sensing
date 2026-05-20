import ast
import datetime
import os
import re
import runpy

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.linalg import qr
from scipy.stats import gaussian_kde, t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_cumulative_sum(data, S, get_user_input_func):
    cumulative_sum = np.cumsum(S)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_sum) + 1), cumulative_sum, marker='o', linestyle='-')
    plt.title('Cumulative Sum of Singular Values')
    plt.xlabel('Number of Singular Values')
    plt.ylabel('Cumulative Sum')
    plt.grid(False)
    r = int(get_user_input_func("Enter r value, press Enter to skip", 3))
    r_cumulative_sum = cumulative_sum[r - 1]
    plt.axvline(x=r, color='r', linestyle='--', label=f'r={r}', ymin=0, ymax=r_cumulative_sum)
    plt.axhline(y=r_cumulative_sum, color='g', linestyle='--', label=f'Cumulative Sum at r={r}', xmin=0, xmax=r)
    plt.annotate(
        f'({r}, {r_cumulative_sum:.2f})',
        xy=(r, r_cumulative_sum),
        xytext=(r + 2, r_cumulative_sum + 20),
        arrowprops=dict(width=0.5, headwidth=5, headlength=5, color='black', shrink=1)
    )
    plt.legend()
    plt.show()


def _option9_event_label(event_file, event_type):
    if event_type == 'design':
        return '200-year designed event'
    match = re.search(r'Flowrates_(\d{1,2})-(\d{1,2})_(\d{4})', str(event_file))
    if not match:
        return str(event_file).replace('Flowrates_', '').replace('.csv', '')
    month, day, year = map(int, match.groups())
    return datetime.date(year, month, day).strftime('%b %d, %Y')


def _extract_time_label(column_name):
    match = re.search(r'(\d{2}:\d{2}:\d{2})', str(column_name))
    return match.group(1) if match else str(column_name)


def _option9_time_ticks(time_labels, interval_hours=4):
    if time_labels:
        tick_hours = list(np.arange(0, 24 + interval_hours, interval_hours, dtype=float))
        tick_labels = [f'{int(hour):02d}:00' if hour < 24 else '24:00' for hour in tick_hours]
        return tick_hours, tick_labels

    parsed_labels = [_extract_time_label(label) for label in time_labels]
    tick_positions = []
    tick_labels = []
    for idx, label in enumerate(parsed_labels, start=1):
        if not re.match(r'\d{2}:\d{2}:\d{2}$', label):
            continue
        hour, minute, second = map(int, label.split(':'))
        if minute == 0 and second == 0 and hour % interval_hours == 0:
            tick_positions.append(idx)
            tick_labels.append(label)
    if not tick_positions and parsed_labels:
        step = max(1, len(parsed_labels) // 6)
        tick_positions = list(range(1, len(parsed_labels) + 1, step))
        tick_labels = [parsed_labels[idx - 1] for idx in tick_positions]
    return tick_positions, tick_labels


def _option9_time_axis(time_labels, n_steps):
    if not time_labels:
        return np.arange(1, n_steps + 1, dtype=float), 1.0, float(n_steps)

    hours = []
    previous = None
    day_offset = 0.0
    for label in time_labels[:n_steps]:
        time_label = _extract_time_label(label)
        if not re.match(r'\d{2}:\d{2}:\d{2}$', time_label):
            return np.arange(1, n_steps + 1, dtype=float), 1.0, float(n_steps)
        hour, minute, second = map(int, time_label.split(':'))
        value = hour + minute / 60.0 + second / 3600.0
        if previous is not None and value <= previous:
            day_offset += 24.0
        hours.append(value + day_offset)
        previous = value
    return np.asarray(hours, dtype=float), 0.0, max(24.0, float(hours[-1]) if hours else 24.0)


def _add_option9_error_bars(ax, time_axis, observed_mean, reconstructed_means, palette, tick_fontsize):
    if observed_mean.size == 0 or len(time_axis) != observed_mean.size:
        return

    observed_peak = float(np.nanmax(observed_mean))
    observed_peak_idx = int(np.nanargmax(observed_mean))
    observed_volume = float(np.nansum(observed_mean))
    duration = float(time_axis[-1] - time_axis[0]) if len(time_axis) > 1 else 1.0
    duration = duration if duration > 0 else 1.0

    metric_labels = ['Qp', 'Tp', 'Vol']
    r_order = [1, 6, 10]
    x_base = np.arange(len(metric_labels), dtype=float)
    width = 0.22

    inset = ax.inset_axes([0.70, 0.39, 0.28, 0.34])
    for offset_idx, r in enumerate(r_order):
        mean_flow = reconstructed_means.get(r)
        if mean_flow is None or len(mean_flow) != observed_mean.size:
            continue
        peak = float(np.nanmax(mean_flow))
        peak_idx = int(np.nanargmax(mean_flow))
        volume = float(np.nansum(mean_flow))
        peak_error = abs(peak - observed_peak) / observed_peak * 100.0 if observed_peak != 0 else 0.0
        time_error = abs(float(time_axis[peak_idx]) - float(time_axis[observed_peak_idx])) / duration * 100.0
        volume_error = abs(volume - observed_volume) / observed_volume * 100.0 if observed_volume != 0 else 0.0
        errors = [peak_error, time_error, volume_error]
        line_color, _ = palette.get(r, ('#4c4c4c', '#bdbdbd'))
        inset.bar(
            x_base + (offset_idx - 1) * width,
            errors,
            width=width,
            color=line_color,
            alpha=0.82,
            edgecolor=line_color,
            linewidth=0.8,
        )
        zero_mask = np.isclose(errors, 0.0)
        if np.any(zero_mask):
            inset.scatter(
                x_base[zero_mask] + (offset_idx - 1) * width,
                np.zeros(np.sum(zero_mask)),
                marker='_',
                s=80,
                color=line_color,
                linewidths=1.3,
                zorder=4,
            )

    inset_fontsize = tick_fontsize
    inset_tick_fontsize = tick_fontsize
    inset.set_xticks(x_base)
    inset.set_xticklabels(metric_labels, fontsize=inset_fontsize, rotation=0, ha='center')
    inset.set_ylabel('Error (%)', fontsize=inset_fontsize, labelpad=2)
    inset.tick_params(axis='y', labelsize=inset_tick_fontsize, length=2)
    inset.tick_params(axis='x', labelsize=inset_fontsize, length=0, pad=4)
    inset.grid(axis='y', alpha=0.18, linewidth=0.5)
    ymax = inset.get_ylim()[1]
    if ymax <= 6:
        step = 2
    elif ymax <= 12:
        step = 4
    else:
        step = 10
    inset.set_yticks(np.arange(0, ymax + step, step))
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor('#555555')


def _plot_shadowline_panel(ax, panel_data, axis_fontsize, tick_fontsize, date_fontsize, panel_tag_fontsize):
    reconstructed_by_r = panel_data['reconstructed_by_r']
    observed_matrix = panel_data['observed_matrix']

    palette = {
        1: ('#1f77b4', '#c6dbef'),
        6: ('#2ca02c', '#c7e9c0'),
        10: ('#d62728', '#fcbba1'),
    }
    marker_styles = {
        'observed': 'o',
        1: 's',
        6: '^',
        10: 'D',
    }
    layer_order = [10, 6, 1]
    line_zorders = {10: 2, 6: 5, 1: 6}
    fill_zorders = {10: 1, 6: 4, 1: 5}

    max_time_steps = 0
    observed_array = np.asarray(observed_matrix, dtype=float).T
    panel_time_axis = None
    panel_xmin = 1.0
    panel_xmax = 1.0
    observed_mean_for_metrics = None
    reconstructed_means_for_metrics = {}
    if observed_array.ndim == 2:
        observed_mean = np.mean(observed_array, axis=1)
        observed_mean_for_metrics = observed_mean
        observed_std = np.std(observed_array, axis=1, ddof=1) if observed_array.shape[1] > 1 else np.zeros(observed_array.shape[0])
        observed_n = observed_array.shape[1]
        observed_t = t.ppf(0.9875, observed_n - 1) if observed_n > 1 else 0.0
        observed_ci = observed_t * observed_std / np.sqrt(observed_n) if observed_n > 1 else np.zeros_like(observed_mean)
        observed_time, panel_xmin, panel_xmax = _option9_time_axis(panel_data.get('time_labels', []), observed_array.shape[0])
        panel_time_axis = observed_time
        observed_markevery = max(1, len(observed_time) // 12)
        ax.plot(
            observed_time,
            observed_mean,
            linewidth=2.4,
            color='#6a51a3',
            linestyle='-',
            marker=marker_styles['observed'],
            markersize=6.6,
            markevery=observed_markevery,
            markerfacecolor='#6a51a3',
            markeredgewidth=1.0,
            label='SWMM-simulated mean',
            zorder=4
        )
        ax.fill_between(
            observed_time,
            observed_mean - observed_ci,
            observed_mean + observed_ci,
            color='#dadaeb',
            alpha=0.32,
            edgecolor='none',
            linewidth=0,
            label='_nolegend_',
            zorder=4
        )
        max_time_steps = max(max_time_steps, observed_array.shape[0])

    for r in layer_order:
        all_x = reconstructed_by_r.get(r)
        if all_x is None:
            continue
        reconstructed_array = np.asarray(all_x, dtype=float)
        if reconstructed_array.ndim != 2:
            continue
        mean_flow = np.mean(reconstructed_array, axis=1)
        reconstructed_means_for_metrics[r] = mean_flow
        std_flow = np.std(reconstructed_array, axis=1, ddof=1) if reconstructed_array.shape[1] > 1 else np.zeros(reconstructed_array.shape[0])
        sample_size = reconstructed_array.shape[1]
        t_value = t.ppf(0.9875, sample_size - 1) if sample_size > 1 else 0.0
        conf_interval = t_value * std_flow / np.sqrt(sample_size) if sample_size > 1 else np.zeros_like(mean_flow)
        if panel_time_axis is not None and len(panel_time_axis) == reconstructed_array.shape[0]:
            time_index = panel_time_axis
        else:
            time_index, panel_xmin, panel_xmax = _option9_time_axis(panel_data.get('time_labels', []), reconstructed_array.shape[0])
        line_color, fill_color = palette.get(r, ('#4c4c4c', '#bdbdbd'))
        markevery = max(1, len(time_index) // 12)

        ax.plot(
            time_index,
            mean_flow,
            linewidth=2.0,
            color=line_color,
            marker=marker_styles.get(r, 'o'),
            markersize=6.2,
            markevery=markevery,
            markerfacecolor=line_color,
            markeredgewidth=0.95,
            label=f'r={r} mean',
            zorder=line_zorders.get(r, 2)
        )
        ax.fill_between(
            time_index,
            mean_flow - conf_interval,
            mean_flow + conf_interval,
            color=fill_color,
            alpha=0.26,
            edgecolor='none',
            linewidth=0,
            label='_nolegend_',
            zorder=fill_zorders.get(r, 1)
        )
        max_time_steps = max(max_time_steps, reconstructed_array.shape[0])

    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_xlim(panel_xmin, panel_xmax)
    if panel_time_axis is not None and observed_mean_for_metrics is not None:
        _add_option9_error_bars(
            ax,
            panel_time_axis,
            observed_mean_for_metrics,
            reconstructed_means_for_metrics,
            palette,
            tick_fontsize,
        )
    panel_box = AnchoredText(
        panel_data['panel_tag'],
        loc='upper left',
        prop=dict(size=panel_tag_fontsize, weight='bold'),
        frameon=False,
        borderpad=0.25,
        pad=0.2,
    )
    ax.add_artist(panel_box)

    event_box = AnchoredText(
        _option9_event_label(panel_data['event_file'], panel_data['event_type']),
        loc='upper right',
        prop=dict(size=date_fontsize),
        frameon=False,
        borderpad=0.25,
        pad=0.2,
    )
    ax.add_artist(event_box)


def plot_shadowline(plot_payload, current_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), dpi=300, sharex=True)

    axis_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 16
    panel_tag_fontsize = 16

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            axis_fontsize = int(input(f"Enter axis label font size (default: {axis_fontsize}): ") or str(axis_fontsize))
            tick_fontsize = int(input(f"Enter tick font size (default: {tick_fontsize}): ") or str(tick_fontsize))
            legend_fontsize = int(input(f"Enter date-label/legend font size (default: {legend_fontsize}): ") or str(legend_fontsize))
            panel_tag_fontsize = int(input(f"Enter panel tag font size (default: {panel_tag_fontsize}): ") or str(panel_tag_fontsize))
        except ValueError:
            print("Invalid input. Using default font sizes.")
            axis_fontsize = 14
            tick_fontsize = 12
            legend_fontsize = 14
            panel_tag_fontsize = 15

    panel_tags = ['a', 'b', 'c']
    for ax, tag, panel_data in zip(np.atleast_1d(axes), panel_tags, plot_payload):
        merged_panel_data = dict(panel_data)
        merged_panel_data['panel_tag'] = tag
        _plot_shadowline_panel(ax, merged_panel_data, axis_fontsize, tick_fontsize, legend_fontsize, panel_tag_fontsize)

    time_labels = plot_payload[0].get('time_labels', []) if plot_payload else []
    tick_positions, tick_labels = _option9_time_ticks(time_labels)
    for ax in axes:
        if tick_positions:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    for ax in axes[:-1]:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    axes[-1].set_xlabel('Time of day', fontsize=axis_fontsize)
    fig.text(0.015, 0.5, 'Flow (CFS)', va='center', rotation='vertical', fontsize=axis_fontsize)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = ['SWMM-simulated mean', 'r=1 mean', 'r=6 mean', 'r=10 mean']
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = [label for label in legend_order if label in handle_by_label]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]
    fig.legend(
        ordered_handles,
        ordered_labels,
        fontsize=legend_fontsize,
        ncol=len(ordered_labels),
        frameon=True,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=[0.035, 0, 1, 0.965])
    plt.savefig(os.path.join(current_dir, 'Shadowline.png'))
    plt.close()


def _draw_half_violin(ax, values, position, color, side='left', width=0.28, alpha=0.52, y_limits=(-4.5, 1.05)):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or np.nanstd(arr) == 0:
        return

    lower = float(np.nanpercentile(arr, 5))
    upper = float(np.nanpercentile(arr, 95))
    y_min = max(y_limits[0], lower - 0.08)
    y_max = min(y_limits[1], upper + 0.08)
    if y_max <= y_min:
        return

    kde_values = arr[(arr >= lower) & (arr <= upper)]
    if kde_values.size < 2 or np.nanstd(kde_values) == 0:
        kde_values = arr

    y_grid = np.linspace(y_min, y_max, 220)
    density = gaussian_kde(kde_values)(y_grid)
    density_max = float(np.nanmax(density))
    if density_max <= 0:
        return

    scaled = density / density_max * width
    if side == 'left':
        ax.fill_betweenx(
            y_grid,
            position,
            position - scaled,
            facecolor=color,
            edgecolor='black',
            linewidth=1.15,
            alpha=alpha,
            zorder=1,
        )
    else:
        ax.fill_betweenx(
            y_grid,
            position,
            position + scaled,
            facecolor=color,
            edgecolor='black',
            linewidth=1.15,
            alpha=alpha,
            zorder=1,
        )


def plot_boxplot(
    nse_values_prediction,
    nse_values_prediction_200,
    r_values,
    current_dir,
    ylabel='NSE',
    output_filename='Reconstructed_x_vs_y.png',
    real_color='#4a98c5',
    design_color='#dc6d57',
    legend_labels=('Real rainfall events', '200-year designed event')
):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'

    r_list = list(r_values)
    use_wide_layout = len(r_list) > 25
    figsize = (14, 8) if use_wide_layout else (12, 8)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    spacing = 2.0
    quartile_info = []
    median_info = []
    bp1 = bp2 = None
    is_nodewise_nse = ylabel == 'Node-level NSE' or output_filename == 'NSE_Boxplot_Per_Node.png'
    for r in r_list:
        real_position = r * spacing - 0.32
        design_position = r * spacing + 0.32
        if is_nodewise_nse:
            _draw_half_violin(ax, nse_values_prediction[r], real_position, real_color, side='left')
            _draw_half_violin(ax, nse_values_prediction_200[r], design_position, design_color, side='right')

        bp1 = ax.boxplot([nse_values_prediction[r]], positions=[real_position], patch_artist=True, widths=0.34 if is_nodewise_nse else 0.6,
                         boxprops={'facecolor': real_color, 'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.78},
                         whiskerprops={'color': 'black', 'linewidth': 1.4},
                         capprops={'color': 'black', 'linewidth': 1.4},
                         flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': real_color,
                                     'markeredgecolor': real_color, 'alpha': 0.75},
                         medianprops={'color': 'black', 'linewidth': 1.5})
        for patch in bp1['boxes']:
            patch.set_facecolor(real_color)

        bp2 = ax.boxplot([nse_values_prediction_200[r]], positions=[design_position], patch_artist=True, widths=0.34 if is_nodewise_nse else 0.6,
                         boxprops={'facecolor': design_color, 'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.78},
                         whiskerprops={'color': 'black', 'linewidth': 1.4},
                         capprops={'color': 'black', 'linewidth': 1.4},
                         flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': design_color,
                                     'markeredgecolor': design_color, 'alpha': 0.75},
                         medianprops={'color': 'black', 'linewidth': 1.5})
        for patch in bp2['boxes']:
            patch.set_facecolor(design_color)

    for r in r_list:
        quartile_info.append(f'r={r}: Q1={np.percentile(nse_values_prediction[r], 25):.3f}, Q3={np.percentile(nse_values_prediction[r], 75):.3f}')
        median_info.append(f'r={r} Real median={np.median(nse_values_prediction[r]):.2f}')
        quartile_info.append(f'r={r}: Q1={np.percentile(nse_values_prediction_200[r], 25):.3f}, Q3={np.percentile(nse_values_prediction_200[r], 75):.3f}')
        median_info.append(f'r={r} 200-year median={np.median(nse_values_prediction_200[r]):.2f}')

    if use_wide_layout:
        tick_r_list = r_list[::2]
    else:
        tick_r_list = r_list
    ax.set_xticks([r * spacing for r in tick_r_list])
    ax.set_xticklabels(tick_r_list)
    ax.set_ylim(-4.5, 1.1)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        list(legend_labels),
        fontsize=14,
        frameon=False,
        facecolor='none'
    )

    x_fontsize = 18
    y_fontsize = 18
    x_tick = 16
    y_tick = 16

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            x_fontsize = int(input(f"Enter x-axis font size (default: {x_fontsize}): ") or str(x_fontsize))
            y_fontsize = int(input(f"Enter y-axis font size (default: {y_fontsize}): ") or str(y_fontsize))
        except ValueError:
            print("Invalid input. Using default font size 18.")
            x_fontsize = 18
            y_fontsize = 18
        tick_choice = input("Also change tick font size? (y/n, default: n): ").strip().lower()
        if tick_choice == 'y':
            x_tick = int(input(f"Enter x-axis tick font size (default: {x_tick}): ") or str(x_tick))
            y_tick = int(input(f"Enter y-axis tick font size (default: {y_tick}): ") or str(y_tick))

    ax.set_xlabel('r (Number of sensors used)', fontsize=x_fontsize)
    ax.set_ylabel(ylabel, fontsize=y_fontsize)
    ax.tick_params(axis='x', labelsize=x_tick)
    ax.tick_params(axis='y', labelsize=y_tick)

    quartile_text = '\n'.join(quartile_info)
    median_text = '\n'.join(median_info)
    choice = input("Display quartile information? (y/n, default: n): ").strip().lower()
    median_choice = input("Display median information? (y/n, default: n): ").strip().lower()
    fig.tight_layout()
    if choice == 'y':
        plt.figtext(0.8, 0.2, quartile_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    if median_choice == 'y':
        plt.figtext(0.6, 0.2, median_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    plt.savefig(os.path.join(current_dir, output_filename))
    plt.close()


def plot_event_nse_linechart(
    nse_values_prediction,
    nse_values_prediction_200,
    r_values,
    current_dir,
    ylabel='System-level NSE',
    output_filename='NSE_Event_Level_Line.png'
):
    r_list = list(r_values)
    use_wide_layout = len(r_list) > 25
    figsize = (14, 8) if use_wide_layout else (12, 8)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    real_event_color = '#4a98c5'
    design_event_color = '#dc6d57'
    r_array = np.asarray(r_list, dtype=float)

    real_means = np.asarray([
        float(np.mean(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_list
    ])
    real_lowers = np.asarray([
        float(np.min(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_list
    ])
    real_uppers = np.asarray([
        float(np.max(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_list
    ])
    design_values = np.asarray([
        float(nse_values_prediction_200[r][0]) if len(nse_values_prediction_200[r]) > 0 else np.nan
        for r in r_list
    ])

    ax.fill_between(
        r_array,
        real_lowers,
        real_uppers,
        color=real_event_color,
        alpha=0.18,
        linewidth=0,
        label='Real rainfall events range'
    )
    ax.plot(
        r_array,
        real_means,
        color=real_event_color,
        linewidth=2.6,
        marker='o',
        markersize=5.5,
        label='Real rainfall events mean'
    )
    ax.plot(
        r_array,
        design_values,
        color=design_event_color,
        linewidth=2.6,
        marker='s',
        markersize=5.2,
        label='200-year designed event'
    )

    ax.set_xlim(r_array.min(), r_array.max())
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(r_array[::2] if use_wide_layout else r_array)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('r (Number of sensors used)', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#4a4a4a')
    ax.legend(fontsize=14, loc='best', frameon=True, edgecolor='#666666')

    fig.tight_layout()
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def _compute_System_Level_Reconstruction_line_stats(nse_values_prediction, nse_values_prediction_200, r_values):
    r_list = list(r_values)
    r_array = np.asarray(r_list, dtype=float)
    real_means = np.asarray([
        float(np.mean(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_list
    ])
    real_lowers = np.asarray([
        float(np.min(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_list
    ])
    real_uppers = np.asarray([
        float(np.max(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_list
    ])
    design_values = np.asarray([
        float(nse_values_prediction_200[r][0]) if len(nse_values_prediction_200[r]) > 0 else np.nan
        for r in r_list
    ])
    return r_list, r_array, real_means, real_lowers, real_uppers, design_values


def _draw_System_Level_Reconstruction_line_panel(ax, nse_values_prediction, nse_values_prediction_200, r_values, axis_fontsize, tick_fontsize, legend_fontsize):
    real_event_color = '#4a98c5'
    design_event_color = '#dc6d57'
    r_list, r_array, real_means, real_lowers, real_uppers, design_values = _compute_System_Level_Reconstruction_line_stats(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values
    )

    ax.fill_between(
        r_array,
        real_lowers,
        real_uppers,
        color=real_event_color,
        alpha=0.18,
        linewidth=0,
        label='Real rainfall events range'
    )
    ax.plot(
        r_array,
        real_means,
        color=real_event_color,
        linewidth=2.8,
        marker='o',
        markersize=6.0,
        label='Real rainfall events mean'
    )
    ax.plot(
        r_array,
        design_values,
        color=design_event_color,
        linewidth=2.8,
        marker='s',
        markersize=5.6,
        label='200-year designed event'
    )

    ax.set_xlim(r_array.min(), r_array.max())
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(r_array[::2] if len(r_list) > 25 else r_array)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel('r (Number of sensors used)', fontsize=axis_fontsize)
    ax.set_ylabel('System-level NSE', fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#4a4a4a')
    ax.legend(fontsize=legend_fontsize, loc='best', frameon=True, edgecolor='#666666')


def _draw_System_Level_Reconstruction_boxplot_panel(ax, nse_values_prediction, nse_values_prediction_200, r_values, axis_fontsize, tick_fontsize, legend_fontsize):
    r_list = list(r_values)
    spacing = 2.0
    for r in r_list:
        bp1 = ax.boxplot(
            [nse_values_prediction[r]],
            positions=[r * spacing - 0.4],
            patch_artist=True,
            widths=0.6,
            flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#4a98c5', 'markeredgecolor': '#4a98c5'},
            medianprops={'color': 'black'}
        )
        for patch in bp1['boxes']:
            patch.set_facecolor('#4a98c5')

        bp2 = ax.boxplot(
            [nse_values_prediction_200[r]],
            positions=[r * spacing + 0.4],
            patch_artist=True,
            widths=0.6,
            flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#dc6d57', 'markeredgecolor': '#dc6d57'},
            medianprops={'color': 'black'}
        )
        for patch in bp2['boxes']:
            patch.set_facecolor('#dc6d57')

    tick_r_list = r_list[::2] if len(r_list) > 25 else r_list
    ax.set_xticks([r * spacing for r in tick_r_list])
    ax.set_xticklabels(tick_r_list)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel('r (Number of sensors used)', fontsize=axis_fontsize)
    ax.set_ylabel('System-level NSE', fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ['Real rainfall events', '200-year designed event'],
        fontsize=legend_fontsize,
        loc='lower right',
        frameon=True,
        edgecolor='#666666'
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#4a4a4a')


def _add_flow_distribution_inset(ax, true_values, pred_values, error_limit):
    """Show compact reconstruction-error boxplot in the scatter-panel whitespace."""
    inset = ax.inset_axes([0.030, 0.56, 0.42, 0.29])
    fill_color = '#4a98c5'
    line_color = '#dc6d57'

    errors = np.asarray(pred_values, dtype=float) - np.asarray(true_values, dtype=float)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        inset.axis('off')
        return

    if not np.isfinite(error_limit) or error_limit <= 0:
        inset.axis('off')
        return

    box = inset.boxplot(
        [errors],
        positions=[0.5],
        widths=0.42,
        patch_artist=True,
        vert=True,
        whis=(5, 95),
        showfliers=False,
        boxprops=dict(facecolor=fill_color, edgecolor=fill_color, alpha=0.30, linewidth=1.0),
        whiskerprops=dict(color=fill_color, linewidth=1.0),
        capprops=dict(color=fill_color, linewidth=1.0),
        medianprops=dict(color=line_color, linewidth=1.5)
    )
    inset.set_xlim(0.15, 0.85)
    inset.set_ylim(-error_limit, error_limit)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor('white')
    inset.axhline(0.0, color='#666666', linewidth=0.8, linestyle='--')
    inset.legend(
        [
            Line2D([0], [0], color=fill_color, linewidth=5, alpha=0.55),
            Line2D([0], [0], color=line_color, linewidth=1.5),
        ],
        ['IQR', 'Median'],
        loc='upper right',
        fontsize=6.5,
        frameon=True,
        borderpad=0.25,
        handlelength=1.5,
        labelspacing=0.2,
        edgecolor='#888888'
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('#777777')


def _draw_System_Level_Reconstruction_scatter_panels(ax_bottom_axes, scatter_datasets, axis_fontsize, tick_fontsize, panel_tag_fontsize, note_fontsize):
    all_true_values = [np.asarray(dataset['true_values']).reshape(-1) for dataset in scatter_datasets]
    all_pred_values = [np.asarray(dataset['pred_values']).reshape(-1) for dataset in scatter_datasets]
    all_errors = [pred - true for true, pred in zip(all_true_values, all_pred_values)]
    combined_min = min(min(np.min(values) for values in all_true_values), min(np.min(values) for values in all_pred_values))
    combined_max = max(max(np.max(values) for values in all_true_values), max(np.max(values) for values in all_pred_values))
    padding = (combined_max - combined_min) * 0.03 if combined_max > combined_min else 1.0
    lower_bound = combined_min - padding
    upper_bound = combined_max + padding
    finite_errors = np.concatenate([err[np.isfinite(err)] for err in all_errors if np.any(np.isfinite(err))])
    if finite_errors.size > 0:
        error_limit = float(np.nanpercentile(np.abs(finite_errors), 99.0))
    else:
        error_limit = 1.0

    last_hex = None
    for idx, dataset in enumerate(scatter_datasets):
        ax = ax_bottom_axes[idx]
        true_values = all_true_values[idx]
        pred_values = all_pred_values[idx]
        last_hex = ax.hexbin(
            true_values,
            pred_values,
            gridsize=52,
            mincnt=1,
            cmap='RdBu_r',
            norm=LogNorm(),
            linewidths=0,
            extent=(lower_bound, upper_bound, lower_bound, upper_bound)
        )
        ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle='--', color='#2f2f2f', linewidth=1.0)
        ax.set_xlim(lower_bound, upper_bound)
        ax.set_ylim(lower_bound, upper_bound)
        ax.set_box_aspect(1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Simulated Flow (CFS)', fontsize=axis_fontsize)
        if idx == 0:
            ax.set_ylabel('Reconstructed Flow (CFS)', fontsize=axis_fontsize)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.grid(True, alpha=0.14, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('#4a4a4a')
        ax.text(
            0.045, 0.955, f'b{idx + 1}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=panel_tag_fontsize,
            fontweight='bold',
        )
        ax.text(
            0.992, 0.012, dataset['label'],
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=note_fontsize,
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.18', linewidth=0.0)
        )

    if len(scatter_datasets) < len(ax_bottom_axes):
        for idx in range(len(scatter_datasets), len(ax_bottom_axes)):
            ax_bottom_axes[idx].axis('off')

    return last_hex


def _plot_System_Level_Reconstruction_combined_core(
    nse_values_prediction,
    nse_values_prediction_200,
    r_values,
    scatter_datasets,
    current_dir,
    output_filename,
    top_kind='line'
):
    fig = plt.figure(figsize=(15, 11.2), dpi=300)
    outer = fig.add_gridspec(
        2, 4,
        height_ratios=[1.65, 1.35],
        width_ratios=[1, 1, 1, 0.04],
        left=0.07, right=0.94, top=0.96, bottom=0.08,
        hspace=0.15, wspace=0.10
    )
    ax_top = fig.add_subplot(outer[0, 0:3])
    ax_bottom_axes = [fig.add_subplot(outer[1, i]) for i in range(3)]
    cax = fig.add_subplot(outer[1, 3])

    title_fontsize = 18
    axis_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 16
    panel_tag_fontsize = 16
    note_fontsize = 16

    font_choice = input("Change combined-figure font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            axis_fontsize = int(input(f"Enter axis label font size (default: {axis_fontsize}): ") or str(axis_fontsize))
            tick_fontsize = int(input(f"Enter tick font size (default: {tick_fontsize}): ") or str(tick_fontsize))
            legend_fontsize = int(input(f"Enter legend font size (default: {legend_fontsize}): ") or str(legend_fontsize))
            panel_tag_fontsize = int(input(f"Enter panel tag font size (default: {panel_tag_fontsize}): ") or str(panel_tag_fontsize))
            note_fontsize = float(input(f"Enter r-label font size (default: {note_fontsize}): ") or str(note_fontsize))
        except ValueError:
            print("Invalid input. Using default font sizes.")

    if top_kind == 'boxplot':
        _draw_System_Level_Reconstruction_boxplot_panel(ax_top, nse_values_prediction, nse_values_prediction_200, r_values, axis_fontsize, tick_fontsize, legend_fontsize)
    else:
        _draw_System_Level_Reconstruction_line_panel(ax_top, nse_values_prediction, nse_values_prediction_200, r_values, axis_fontsize, tick_fontsize, legend_fontsize)

    ax_top.text(
        0.015, 0.955, 'a',
        transform=ax_top.transAxes,
        ha='left',
        va='top',
        fontsize=title_fontsize,
        fontweight='bold',
    )

    last_hex = _draw_System_Level_Reconstruction_scatter_panels(
        ax_bottom_axes,
        scatter_datasets,
        axis_fontsize,
        tick_fontsize,
        panel_tag_fontsize,
        note_fontsize
    )

    if last_hex is not None:
        cbar = fig.colorbar(last_hex, cax=cax)
        cbar.set_label('Counts', fontsize=legend_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
    else:
        cax.axis('off')

    fig.tight_layout()
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_System_Level_Reconstruction_combined_boxplot_scatter(
    nse_values_prediction,
    nse_values_prediction_200,
    r_values,
    scatter_datasets,
    current_dir,
    output_filename='System_Level_Reconstruction_Combined_Boxplot_Scatter.png'
):
    return _plot_System_Level_Reconstruction_combined_core(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        scatter_datasets,
        current_dir,
        output_filename,
        top_kind='boxplot'
    )


def plot_System_Level_Reconstruction_combined_line_scatter(
    nse_values_prediction,
    nse_values_prediction_200,
    r_values,
    scatter_datasets,
    current_dir,
    output_filename='System_Level_Reconstruction_Combined_Line_Scatter.png'
):
    return _plot_System_Level_Reconstruction_combined_core(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        scatter_datasets,
        current_dir,
        output_filename,
        top_kind='line'
    )


def plot_System_Level_Reconstruction_combined_figure(
    nse_values_prediction,
    nse_values_prediction_200,
    r_values,
    scatter_datasets,
    current_dir,
    output_filename='System_Level_Reconstruction_Combined_Boxplot_Scatter.png'
):
    fig = plt.figure(figsize=(15, 11.2), dpi=300)
    outer = fig.add_gridspec(
        2, 4,
        height_ratios=[1.65, 1.35],
        width_ratios=[1, 1, 1, 0.04],
        left=0.07, right=0.94, top=0.96, bottom=0.08,
        hspace=0.15, wspace=0.10
    )
    ax_top = fig.add_subplot(outer[0, 0:3])
    ax_bottom_axes = [fig.add_subplot(outer[1, i]) for i in range(3)]
    cax = fig.add_subplot(outer[1, 3])

    title_fontsize = 18
    axis_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 16
    panel_tag_fontsize = 16
    note_fontsize = 16

    font_choice = input("Change combined-figure font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            axis_fontsize = int(input(f"Enter axis label font size (default: {axis_fontsize}): ") or str(axis_fontsize))
            tick_fontsize = int(input(f"Enter tick font size (default: {tick_fontsize}): ") or str(tick_fontsize))
            legend_fontsize = int(input(f"Enter legend/colorbar font size (default: {legend_fontsize}): ") or str(legend_fontsize))
            panel_tag_fontsize = int(input(f"Enter panel tag font size (default: {panel_tag_fontsize}): ") or str(panel_tag_fontsize))
            note_fontsize = float(input(f"Enter r-label font size (default: {note_fontsize}): ") or str(note_fontsize))
        except ValueError:
            print("Invalid input. Using default font sizes.")

    real_event_color = '#4a98c5'
    design_event_color = '#dc6d57'
    r_array = np.asarray(list(r_values), dtype=float)
    real_means = np.asarray([
        float(np.mean(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_values
    ])
    real_lowers = np.asarray([
        float(np.min(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_values
    ])
    real_uppers = np.asarray([
        float(np.max(nse_values_prediction[r])) if len(nse_values_prediction[r]) > 0 else np.nan
        for r in r_values
    ])
    design_values = np.asarray([
        float(nse_values_prediction_200[r][0]) if len(nse_values_prediction_200[r]) > 0 else np.nan
        for r in r_values
    ])

    ax_top.fill_between(
        r_array,
        real_lowers,
        real_uppers,
        color=real_event_color,
        alpha=0.18,
        linewidth=0,
        label='Real rainfall events range'
    )
    ax_top.plot(
        r_array,
        real_means,
        color=real_event_color,
        linewidth=2.8,
        marker='o',
        markersize=6.0,
        label='Real rainfall events mean'
    )
    ax_top.plot(
        r_array,
        design_values,
        color=design_event_color,
        linewidth=2.8,
        marker='s',
        markersize=5.6,
        label='200-year designed event'
    )

    ax_top.set_xlim(r_array.min(), r_array.max())
    ax_top.set_xticks(r_array)
    ax_top.set_xticklabels([int(r) for r in r_array])
    ax_top.set_ylim(0.7, 1.0)
    ax_top.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_top.set_xlabel('r (Number of sensors used)', fontsize=axis_fontsize)
    ax_top.set_ylabel('System-level NSE', fontsize=axis_fontsize)
    ax_top.tick_params(axis='both', labelsize=tick_fontsize)
    ax_top.grid(axis='y', alpha=0.16, linewidth=0.8)
    for spine in ax_top.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#4a4a4a')
    ax_top.legend(
        fontsize=legend_fontsize,
        loc='lower right',
        frameon=False,
        facecolor='none'
    )
    ax_top.text(
        0.015, 0.955, 'a',
        transform=ax_top.transAxes,
        ha='left',
        va='top',
        fontsize=title_fontsize,
        fontweight='bold',
    )

    all_true_values = [np.asarray(dataset['true_values']).reshape(-1) for dataset in scatter_datasets]
    all_pred_values = [np.asarray(dataset['pred_values']).reshape(-1) for dataset in scatter_datasets]
    all_errors = [pred - true for true, pred in zip(all_true_values, all_pred_values)]
    combined_min = min(min(np.min(values) for values in all_true_values), min(np.min(values) for values in all_pred_values))
    combined_max = max(max(np.max(values) for values in all_true_values), max(np.max(values) for values in all_pred_values))
    padding = (combined_max - combined_min) * 0.03 if combined_max > combined_min else 1.0
    lower_bound = combined_min - padding
    upper_bound = combined_max + padding
    finite_errors = np.concatenate([err[np.isfinite(err)] for err in all_errors if np.any(np.isfinite(err))])
    if finite_errors.size > 0:
        error_limit = float(np.nanpercentile(np.abs(finite_errors), 99.0))
    else:
        error_limit = 1.0

    last_hex = None
    for idx, dataset in enumerate(scatter_datasets):
        ax = ax_bottom_axes[idx]
        true_values = all_true_values[idx]
        pred_values = all_pred_values[idx]
        last_hex = ax.hexbin(
            true_values,
            pred_values,
            gridsize=52,
            mincnt=1,
            cmap='RdBu_r',
            norm=LogNorm(),
            linewidths=0,
            extent=(lower_bound, upper_bound, lower_bound, upper_bound)
        )
        ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle='--', color='#2f2f2f', linewidth=1.0)
        ax.set_xlim(lower_bound, upper_bound)
        ax.set_ylim(lower_bound, upper_bound)
        ax.set_box_aspect(1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Simulated Flow (CFS)', fontsize=axis_fontsize)
        if idx == 0:
            ax.set_ylabel('Reconstructed Flow (CFS)', fontsize=axis_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.grid(True, alpha=0.14, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('#4a4a4a')
        if idx > 0:
            ax.set_ylabel('')
        if idx > 0:
            ax.tick_params(axis='y', labelleft=False)
        ax.text(
            0.045, 0.955, f'b{idx + 1}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=panel_tag_fontsize,
            fontweight='bold',
        )
        ax.text(
            0.992, 0.012, dataset['label'],
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=note_fontsize,
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.18', linewidth=0.0)
        )

    if len(scatter_datasets) < len(ax_bottom_axes):
        for idx in range(len(scatter_datasets), len(ax_bottom_axes)):
            ax_bottom_axes[idx].axis('off')

    if last_hex is not None:
        cbar = fig.colorbar(
            last_hex,
            cax=cax
        )
        cbar.set_label('Counts', fontsize=legend_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('#666666')
        left_ax_pos = ax_bottom_axes[0].get_position()
        cbar_pos = cax.get_position()
        cax.set_position([
            cbar_pos.x0 - 0.006,
            left_ax_pos.y0,
            cbar_pos.width * 1.18,
            left_ax_pos.height
        ])

    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_boxplot_Noisy(nse_values_prediction, nse_values_prediction_noisy_5, nse_values_prediction_noisy_10,
                       nse_values_prediction_noisy_15, r_values, current_dir, ylabel='NSE',
                       output_filename='Reconstructed x vs y Noisy.png'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    spacing = 4.0
    quartile_info = []
    for r in r_values:
        positions = [r * spacing - 1.2, r * spacing - 0.4, r * spacing + 0.4, r * spacing + 1.2]
        datasets = [
            (nse_values_prediction[r], '#89C0B7'),
            (nse_values_prediction_noisy_5[r], '#B7E1E4'),
            (nse_values_prediction_noisy_10[r], '#6F91B5'),
            (nse_values_prediction_noisy_15[r], '#EF8F88')
        ]
        boxes = []
        for idx, (vals, color) in enumerate(datasets):
            bp = ax.boxplot([vals], positions=[positions[idx]], patch_artist=True, widths=0.6,
                            flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': color, 'markeredgecolor': color},
                            medianprops={'color': 'black'})
            for patch in bp['boxes']:
                patch.set_facecolor(color)
            boxes.append(bp)
            quartile_info.append(f'r={r}: Q1={np.percentile(vals, 25):.3f}, Q3={np.percentile(vals, 75):.3f}')
        bp1, bp2, bp3, bp4 = boxes

    ax.set_xticks([r * spacing for r in r_values])
    ax.set_xticklabels(r_values)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    # Prefer an in-axes empty area for the legend before using any external placement.
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0]],
        ['Noise-free', '5% Gaussian Noise', '10% Gaussian Noise', '15% Gaussian Noise'],
        fontsize=12,
        loc='best',
        frameon=True,
    )

    if input("Add chart title describing noise model? (y/n, default: n): ").strip().lower() == 'y':
        ax.set_title('NSE Performance Under Multiplicative Gaussian Noise (Monte Carlo Analysis)', fontsize=18, pad=20)

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            x_fontsize = int(input("Enter x-axis font size (default: 16): ") or "18")
            y_fontsize = int(input("Enter y-axis font size (default: 16): ") or "18")
        except ValueError:
            print("Invalid input. Using default font size 18.")
            x_fontsize = 18
            y_fontsize = 18
        ax.set_xlabel('r (Number of sensors used)', fontsize=x_fontsize)
        ax.set_ylabel(ylabel, fontsize=y_fontsize)
        tick_choice = input("Also change tick font size? (y/n, default: n): ").strip().lower()
        if tick_choice == 'y':
            x_tick = int(input("Enter x-axis tick font size (default: 16): ") or "16")
            y_tick = int(input("Enter y-axis tick font size (default: 16): ") or "16")
            ax.tick_params(axis='x', labelsize=x_tick)
            ax.tick_params(axis='y', labelsize=y_tick)
    else:
        ax.set_xlabel('r (Number of sensors used)')
        ax.set_ylabel(ylabel)

    quartile_text = '\n'.join(quartile_info)
    fig.tight_layout()
    if input("Display quartile information? (y/n, default: n): ").strip().lower() == 'y':
        plt.figtext(0.6, 0.1, quartile_text, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}, fontsize=8)
    plt.savefig(os.path.join(current_dir, output_filename))
    plt.close()


def _draw_noisy_boxplot_panel(
    ax,
    clean_values_by_r,
    noisy_5_by_r,
    noisy_10_by_r,
    noisy_15_by_r,
    r_values,
    ylabel,
    panel_tag,
    box_width=0.6
):
    spacing = 4.0
    color_pairs = [
        (clean_values_by_r, '#89C0B7'),
        (noisy_5_by_r, '#B7E1E4'),
        (noisy_10_by_r, '#6F91B5'),
        (noisy_15_by_r, '#EF8F88'),
    ]

    for r in r_values:
        positions = [r * spacing - 1.2, r * spacing - 0.4, r * spacing + 0.4, r * spacing + 1.2]
        boxes = []
        for idx, (dataset_map, color) in enumerate(color_pairs):
            vals = dataset_map[r]
            bp = ax.boxplot(
                [vals],
                positions=[positions[idx]],
                patch_artist=True,
                widths=box_width,
                flierprops={'marker': 'o', 'markersize': 2.8, 'markerfacecolor': color, 'markeredgecolor': color},
                boxprops={'edgecolor': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.4},
                capprops={'color': 'black', 'linewidth': 1.4},
                medianprops={'color': 'black', 'linewidth': 1.5}
            )
            for patch in bp['boxes']:
                patch.set_facecolor(color)
            boxes.append(bp)
        bp1, bp2, bp3, bp4 = boxes

    ax.set_xticks([r * spacing for r in r_values])
    ax.set_xticklabels(r_values)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', alpha=0.12, linewidth=0.8)
    ax.text(
        0.015, 0.965, panel_tag,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=18,
        fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='#444444', boxstyle='square,pad=0.22', linewidth=0.9)
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#4a4a4a')

    return bp1, bp2, bp3, bp4


def plot_noisy_dual_boxplot(
    node_clean_by_r,
    node_noisy_5_by_r,
    node_noisy_10_by_r,
    node_noisy_15_by_r,
    event_clean_by_r,
    event_noisy_5_by_r,
    event_noisy_10_by_r,
    event_noisy_15_by_r,
    r_values,
    current_dir,
    output_filename='Noisy_NSE_Node_Event_Combined.png'
):
    fig, axes = plt.subplots(2, 1, figsize=(13, 13), dpi=300, sharex=True)

    top_boxes = _draw_noisy_boxplot_panel(
        axes[0],
        event_clean_by_r,
        event_noisy_5_by_r,
        event_noisy_10_by_r,
        event_noisy_15_by_r,
        r_values,
        ylabel='System-level NSE',
        panel_tag='A'
    )
    _draw_noisy_boxplot_panel(
        axes[1],
        node_clean_by_r,
        node_noisy_5_by_r,
        node_noisy_10_by_r,
        node_noisy_15_by_r,
        r_values,
        ylabel='Node-level NSE',
        panel_tag='B'
    )

    axes[1].set_xlabel('r (Number of sensors used)', fontsize=18)
    axes[0].legend(
        [top_boxes[0]["boxes"][0], top_boxes[1]["boxes"][0], top_boxes[2]["boxes"][0], top_boxes[3]["boxes"][0]],
        ['Noise-free', '5% Gaussian Noise', '10% Gaussian Noise', '15% Gaussian Noise'],
        fontsize=13,
        loc='best',
        frameon=True,
        edgecolor='#666666'
    )

    fig.tight_layout()
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_noisy_system_boxplot(
    event_clean_by_r,
    event_noisy_5_by_r,
    event_noisy_10_by_r,
    event_noisy_15_by_r,
    r_values,
    current_dir,
    output_filename='Reconstructed_x_vs_y_Noisy_System_Level_NSE.png'
):
    fig, ax = plt.subplots(figsize=(13, 6.8), dpi=300)

    top_boxes = _draw_noisy_boxplot_panel(
        ax,
        event_clean_by_r,
        event_noisy_5_by_r,
        event_noisy_10_by_r,
        event_noisy_15_by_r,
        r_values,
        ylabel='System-level NSE',
        panel_tag=None,
        box_width=0.48
    )

    ax.set_xlabel('r (Number of sensors used)', fontsize=18)
    ax.set_ylim(0.70, 1.01)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.legend(
        [top_boxes[0]["boxes"][0], top_boxes[1]["boxes"][0], top_boxes[2]["boxes"][0], top_boxes[3]["boxes"][0]],
        ['Noise-free', '5% Gaussian Noise', '10% Gaussian Noise', '15% Gaussian Noise'],
        fontsize=14,
        loc='lower right',
        frameon=False,
        facecolor='none'
    )

    fig.tight_layout()
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_boxplot_SD(nse_values_prediction_d, event_nse_prediction_d, r_value, Node_ID_D, current_dir, ylabel='NSE',
                    output_filename='Reconstructed x vs y SD.png'):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    spacing = 1.5
    quartile_info = []
    for r in range(0, r_value + 1):
        bp1 = ax.boxplot([nse_values_prediction_d[r]], positions=[r * spacing - 0.2], patch_artist=True, widths=0.35,
                         flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#afd4e3',
                                     'markeredgecolor': '#afd4e3'}, medianprops={'color': 'black'})
        for patch in bp1['boxes']:
            patch.set_facecolor('#afd4e3')
        bp2 = ax.boxplot([event_nse_prediction_d[r]], positions=[r * spacing + 0.2], patch_artist=True, widths=0.35,
                         flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#dc6d57',
                                     'markeredgecolor': '#dc6d57'}, medianprops={'color': 'black'})
        for patch in bp2['boxes']:
            patch.set_facecolor('#dc6d57')
        quartile_info.append(f'Node-level {r}: Q1={np.percentile(nse_values_prediction_d[r], 25):.3f}, Q3={np.percentile(nse_values_prediction_d[r], 75):.3f}')
        quartile_info.append(f'System-level {r}: Q1={np.percentile(event_nse_prediction_d[r], 25):.3f}, Q3={np.percentile(event_nse_prediction_d[r], 75):.3f}')

    ax.set_xticks([r * spacing for r in range(0, r_value + 1)])
    sorted_keys = sorted(Node_ID_D.keys())
    sorted_labels = [Node_ID_D[key] for key in sorted_keys]
    ax.set_xticklabels(sorted_labels)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('r')
    ax.set_ylabel(ylabel)
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ['Node-level NSE', 'System-level NSE'],
        fontsize=12,
        loc='lower right',
        frameon=True,
        edgecolor='#666666'
    )

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            x_fontsize = int(input("Enter x-axis font size (default: 14): ") or "14")
            y_fontsize = int(input("Enter y-axis font size (default: 14): ") or "14")
        except ValueError:
            print("Invalid input. Using default font size 12.")
            x_fontsize = 14
            y_fontsize = 14
        ax.set_xlabel('Name of the dropped sensor', fontsize=x_fontsize)
        ax.set_ylabel(ylabel, fontsize=y_fontsize)
        tick_choice = input("Also change tick font size? (y/n, default: n): ").strip().lower()
        if tick_choice == 'y':
            x_tick = int(input("Enter x-axis tick font size (default: 12): ") or "12")
            y_tick = int(input("Enter y-axis tick font size (default: 12): ") or "12")
            ax.tick_params(axis='x', labelsize=x_tick)
            ax.tick_params(axis='y', labelsize=y_tick)
    else:
        ax.set_xlabel('Name of the dropped sensor')
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    if input("Display quartile information? (y/n, default: n): ").strip().lower() == 'y':
        plt.figtext(0.7, 0.1, '\n'.join(quartile_info), bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    plt.savefig(os.path.join(current_dir, output_filename))
    plt.close()


def plot_boxplot_random(nse_values, nse_values_random, r_values, current_dir):
    all_nse_values = []
    all_nse_values_random = []
    quartile_info = []
    for r in r_values:
        all_nse_values.extend(nse_values[r])
        for iteration in nse_values_random[r]:
            all_nse_values_random.extend(iteration)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True, dpi=300)
    fig.subplots_adjust(hspace=0.05)
    bp1_upper = ax1.boxplot([all_nse_values], positions=[1], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor='#EBD8B7', color='black'),
                            medianprops=dict(color='black'),
                            flierprops=dict(marker='o', markersize=4, markerfacecolor='#EBD8B7', markeredgecolor='#EBD8B7'))
    quartile_info.append(f'Optimization: Q1={np.percentile(all_nse_values, 25):.3f}, Q3={np.percentile(all_nse_values, 75):.3f}')
    bp2_upper = ax1.boxplot([all_nse_values_random], positions=[2], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor='#6C85A4', color='black'),
                            medianprops=dict(color='black'),
                            flierprops=dict(marker='o', markersize=4, markerfacecolor='#6C85A4', markeredgecolor='#6C85A4'))
    quartile_info.append(f'Random: Q1={np.percentile(all_nse_values_random, 25):.3f}, Q3={np.percentile(all_nse_values_random, 75):.3f}')
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(top=False, labeltop=False)

    ax2.boxplot([all_nse_values], positions=[1], widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='#EBD8B7', color='black'),
                medianprops=dict(color='black'),
                flierprops=dict(marker='o', markersize=4, markerfacecolor='#EBD8B7', markeredgecolor='#EBD8B7'))
    ax2.boxplot([all_nse_values_random], positions=[2], widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='#6C85A4', color='black'),
                medianprops=dict(color='black'),
                flierprops=dict(marker='o', markersize=4, markerfacecolor='#6C85A4', markeredgecolor='#6C85A4'))
    ax2.set_ylim(-5, -0.5)
    ax2.set_yticks([-5, -4, -3, -2, -1])
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(bottom=False, labelbottom=False)

    bp1_lower = ax3.boxplot([all_nse_values], positions=[1], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor='#EBD8B7', color='black'),
                            medianprops=dict(color='black'),
                            flierprops=dict(marker='o', markersize=4, markerfacecolor='#EBD8B7', markeredgecolor='#EBD8B7'))
    bp2_lower = ax3.boxplot([all_nse_values_random], positions=[2], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor='#6C85A4', color='black'),
                            medianprops=dict(color='black'),
                            flierprops=dict(marker='o', markersize=4, markerfacecolor='#6C85A4', markeredgecolor='#6C85A4'))
    ax3.set_ylim(-100, -10)
    ax3.set_yticks([-100, -80, -60, -40, -20])
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.tick_bottom()
    ax3.tick_params(labelbottom=False)

    d = .012
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax3.transAxes)
    ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Optimized', 'Random'])
    ax3.tick_params(axis='x', labelbottom=True)
    ax3.legend([bp1_lower["boxes"][0], bp2_lower["boxes"][0]], ['Optimized', 'Random'], fontsize=14, loc='lower left')

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            x_fontsize = int(input("Enter x-axis font size (default: 16): ") or "16")
            y_fontsize = int(input("Enter y-axis font size (default: 16): ") or "16")
        except ValueError:
            print("Invalid input. Using default font size 14.")
            x_fontsize = 16
            y_fontsize = 16
        ax3.set_xlabel('Placement Method', fontsize=x_fontsize)
        fig.text(0.02, 0.5, 'NSE', va='center', rotation='vertical', fontsize=y_fontsize)
        tick_choice = input("Also change tick font size? (y/n, default: n): ").strip().lower()
        if tick_choice == 'y':
            x_tick = int(input("Enter x-axis tick font size (default: 14): ") or "14")
            y_tick = int(input("Enter y-axis tick font size (default: 14): ") or "14")
            ax3.tick_params(axis='x', labelsize=x_tick)
            ax1.tick_params(axis='y', labelsize=y_tick)
            ax2.tick_params(axis='y', labelsize=y_tick)
            ax3.tick_params(axis='y', labelsize=y_tick)
    else:
        ax3.set_xlabel('Placement Method')
        fig.text(0.02, 0.5, 'NSE', va='center', rotation='vertical', fontsize=14)

    fig.tight_layout()
    if input("Display quartile information? (y/n, default: n): ").strip().lower() == 'y':
        plt.savefig(os.path.join(current_dir, 'Reconstructed x vs y_random.png'))
        plt.figtext(0.5, 0.1, '\n'.join(quartile_info), bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.savefig(os.path.join(current_dir, 'Reconstructed x vs y_random_q.png'))
    else:
        plt.savefig(os.path.join(current_dir, 'Reconstructed x vs y_random.png'))
    plt.close()


def _flatten_series_dict(series_by_r):
    values = []
    for series in series_by_r.values():
        values.extend(list(series))
    return np.asarray(values, dtype=float) if values else np.asarray([], dtype=float)


def _event_offsets(n_points, center, width=0.24):
    if n_points <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - width / 2, center + width / 2, n_points)


def _boxplot_whisker_bounds(distributions):
    lower_bounds = []
    upper_bounds = []
    for values in distributions.values():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        inliers = arr[(arr >= lower_fence) & (arr <= upper_fence)]
        if inliers.size == 0:
            inliers = arr
        lower_bounds.append(float(np.min(inliers)))
        upper_bounds.append(float(np.max(inliers)))
    if not lower_bounds:
        return 0.0, 1.0
    return min(lower_bounds), max(upper_bounds)


def _broken_axis_limits(distribution_by_r, overlay_by_r_list):
    distribution_values = _flatten_series_dict(distribution_by_r)
    overlay_values = []
    for overlay_by_r in overlay_by_r_list:
        overlay_values.extend(_flatten_series_dict(overlay_by_r).tolist())
    overlay_values = np.asarray(overlay_values, dtype=float) if overlay_values else np.asarray([], dtype=float)

    if distribution_values.size == 0:
        return [(0.0, 1.0)]

    finite_distribution = distribution_values[np.isfinite(distribution_values)]
    if finite_distribution.size == 0:
        return [(0.0, 1.0)]

    overlay_finite = overlay_values[np.isfinite(overlay_values)] if overlay_values.size else np.asarray([], dtype=float)
    combined = finite_distribution if overlay_finite.size == 0 else np.concatenate([finite_distribution, overlay_finite])

    q1 = float(np.percentile(finite_distribution, 25))
    q3 = float(np.percentile(finite_distribution, 75))
    lower_whisker, upper_whisker = _boxplot_whisker_bounds(distribution_by_r)
    overlay_max = float(np.max(overlay_finite)) if overlay_finite.size else upper_whisker

    top_low = max(0.75, min(0.88, q3 - 0.03))
    top_high = min(1.02, max(1.0, overlay_max + 0.015))
    top_limits = (top_low, top_high)

    mid_high = top_low - 0.02
    mid_low = min(-0.1, q1 - 0.1 * max(q3 - q1, 0.5))
    mid_low = max(mid_low, lower_whisker + 0.15 * max(abs(lower_whisker), 1.0))
    if mid_low >= mid_high:
        mid_low = min(-0.2, q1 - 0.4)
    mid_limits = (mid_low, mid_high)

    lower_high = min(-5.0, lower_whisker + 0.1 * max(abs(lower_whisker), 1.0))
    lower_low = lower_whisker - 0.15 * max(abs(lower_whisker), 5.0)
    if lower_low >= lower_high:
        lower_low = lower_high - 5.0
    lower_limits = (lower_low, lower_high)

    return [top_limits, mid_limits, lower_limits]


def _draw_distribution_boxes(ax, positions, distributions, color, label=None, widths=0.5):
    box = ax.boxplot(
        distributions,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=color, color='black', alpha=0.75, linewidth=1.5),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.4),
        capprops=dict(color='black', linewidth=1.4),
        flierprops=dict(marker='o', markersize=3, markerfacecolor=color, markeredgecolor=color, alpha=0.55),
    )
    if label:
        box['boxes'][0].set_label(label)
    return box


def _boxplot_stats_percentile(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1 = float(np.percentile(arr, 25))
    median = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    whislo = float(np.percentile(arr, 5))
    whishi = float(np.percentile(arr, 95))
    return {
        'med': median,
        'q1': q1,
        'q3': q3,
        'whislo': whislo,
        'whishi': whishi,
        'fliers': [],
    }


def _draw_distribution_boxes_percentile(ax, positions, distributions, color, label=None, widths=0.5):
    stats = []
    for pos, values in zip(positions, distributions):
        stat = _boxplot_stats_percentile(values)
        if stat is None:
            continue
        stat['label'] = str(pos)
        stats.append(stat)
    box = ax.bxp(
        stats,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=color, color='black', alpha=0.75),
        medianprops=dict(color='black', linewidth=1.4),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )
    if label:
        box['boxes'][0].set_label(label)
    return box


def _style_broken_axis_stack(axes, show_xlabel=False, xlabel='Number of Sensors (r)', label_size=12, tick_size=11):
    d = 0.012
    for idx, ax in enumerate(axes):
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.grid(False)
        ax.tick_params(axis='y', labelsize=tick_size)
        if idx == 0:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.tick_top()
            ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
        elif idx < len(axes) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
        else:
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', labelsize=tick_size)
            if show_xlabel:
                ax.set_xlabel(xlabel, fontsize=label_size)

    if len(axes) > 1:
        for upper_ax, lower_ax in zip(axes[:-1], axes[1:]):
            kwargs = dict(transform=upper_ax.transAxes, color='k', clip_on=False, linewidth=1)
            upper_ax.plot((-d, +d), (-d, +d), **kwargs)
            upper_ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=lower_ax.transAxes)
            lower_ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            lower_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def _draw_event_points(ax, positions, series_by_r, color, marker, label=None, offset_width=0.24):
    first = True
    for x_pos, r in zip(positions, sorted(series_by_r.keys())):
        event_values = [float(value) for value in series_by_r[r] if np.isfinite(value)]
        if not event_values:
            continue
        offsets = _event_offsets(len(event_values), x_pos, width=offset_width)
        ax.scatter(
            offsets,
            event_values,
            color=color,
            s=28,
            marker=marker,
            zorder=5,
            label=label if first else None,
        )
        median_value = float(np.median(event_values))
        ax.hlines(median_value, x_pos - 0.12, x_pos + 0.12, colors=color, linewidth=2.2, zorder=6)
        first = False


def plot_exhaustive_eventwise_benchmark(
    exhaustive_distribution_by_r,
    dss_eventwise_by_r,
    optimum_eventwise_by_r,
    current_dir,
    output_filename='Exhaustive_benchmark_eventwise.png',
    title_suffix='',
    axis_segments=None,
    show_panel_labels=False,
):
    label_size = 13
    tick_size = 12
    r_values = sorted(exhaustive_distribution_by_r.keys())
    if not r_values:
        return

    distribution_positions = np.arange(1, len(r_values) + 1, dtype=float)
    dss_positions = distribution_positions - 0.28
    optimum_positions = distribution_positions + 0.28
    axis_segments = axis_segments or _broken_axis_limits(exhaustive_distribution_by_r, [dss_eventwise_by_r, optimum_eventwise_by_r])

    fig = plt.figure(figsize=(9, 10), dpi=300)
    grid = GridSpec(len(axis_segments) + 1, 1, height_ratios=[1] * len(axis_segments) + [1.0], hspace=0.05)
    dist_axes = [fig.add_subplot(grid[idx, 0]) for idx in range(len(axis_segments))]
    gap_ax = fig.add_subplot(grid[len(axis_segments), 0], sharex=dist_axes[-1])

    distributions = [exhaustive_distribution_by_r[r] for r in r_values]
    gap_values = {
        r: [opt - dss for opt, dss in zip(optimum_eventwise_by_r[r], dss_eventwise_by_r[r])]
        for r in r_values
    }

    box_handle = None
    for idx, ax in enumerate(dist_axes):
        box = _draw_distribution_boxes(ax, distribution_positions, distributions, '#3B5DA3', label='All combinations', widths=0.28)
        if box_handle is None:
            box_handle = box['boxes'][0]
        _draw_event_points(ax, dss_positions, dss_eventwise_by_r, '#f8dfa6', 'o', label='DSS')
        _draw_event_points(ax, optimum_positions, optimum_eventwise_by_r, '#6CB3DA', 's', label='Exhaustive optimum')
        ax.set_ylim(*axis_segments[idx])

    _style_broken_axis_stack(dist_axes, show_xlabel=True)
    dist_axes[-1].set_xticks(distribution_positions)
    dist_axes[-1].set_xticklabels([str(r) for r in r_values])

    top_stack_center = (dist_axes[0].get_position().y1 + dist_axes[-1].get_position().y0) / 2
    fig.text(0.04, top_stack_center, 'System-level NSE', va='center', rotation='vertical', fontsize=label_size)
    if show_panel_labels:
        dist_axes[0].text(0.015, 0.96, '(a)', transform=dist_axes[0].transAxes, fontsize=14, fontweight='bold', va='top')

    gap_distributions = [gap_values[r] for r in r_values]
    gap_box = _draw_distribution_boxes(gap_ax, distribution_positions, gap_distributions, '#84c1ce', label='Optimum - DSS', widths=0.26)
    _draw_event_points(gap_ax, distribution_positions, gap_values, '#dea397', 'D', label='System-level gap')
    gap_ax.axhline(0.0, color='black', linewidth=1.0, linestyle='--')
    gap_ax.set_xticks(distribution_positions)
    gap_ax.set_xticklabels([str(r) for r in r_values])
    gap_ax.tick_params(axis='both', labelsize=tick_size)
    gap_ax.set_xlabel('Number of Sensors (r)', fontsize=label_size)
    gap_ax.set_ylabel('NSE Gap', fontsize=label_size)
    gap_ax.spines['right'].set_visible(True)
    gap_ax.spines['left'].set_visible(True)
    gap_ax.yaxis.set_label_coords(-0.085, 0.5)
    if show_panel_labels:
        gap_ax.text(0.015, 0.96, '(b)', transform=gap_ax.transAxes, fontsize=14, fontweight='bold', va='top')
    gap_ax.legend(
        [
            box_handle,
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f8dfa6', markeredgecolor='#f8dfa6', markersize=7),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#6CB3DA', markeredgecolor='#6CB3DA', markersize=7),
            gap_box['boxes'][0],
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#dea397', markeredgecolor='#dea397', markersize=7),
        ],
        ['All combinations', 'DSS', 'Exhaustive optimum', 'Optimum - DSS', 'System-level gap'],
        fontsize=9.4,
        loc='upper right',
        frameon=True,
        ncol=2,
        columnspacing=0.72,
        handletextpad=0.42,
        borderpad=0.28
    )

    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.18, hspace=0.05)
    plt.savefig(os.path.join(current_dir, output_filename), bbox_inches='tight')
    plt.close(fig)


def plot_exhaustive_eventwise_benchmark_percentile(
    exhaustive_distribution_by_r,
    dss_eventwise_by_r,
    optimum_eventwise_by_r,
    current_dir,
    output_filename='Exhaustive_benchmark_eventwise_percentile.png',
    title_suffix='',
    axis_segments=None,
):
    title_size = 13
    label_size = 12
    tick_size = 11
    r_values = sorted(exhaustive_distribution_by_r.keys())
    if not r_values:
        return

    distribution_positions = np.arange(1, len(r_values) + 1, dtype=float)
    dss_positions = distribution_positions - 0.28
    optimum_positions = distribution_positions + 0.28
    axis_segments = axis_segments or _broken_axis_limits(exhaustive_distribution_by_r, [dss_eventwise_by_r, optimum_eventwise_by_r])

    fig = plt.figure(figsize=(9, 8), dpi=300)
    grid = GridSpec(len(axis_segments), 1, height_ratios=[1] * len(axis_segments), hspace=0.05)
    axes = [fig.add_subplot(grid[idx, 0]) for idx in range(len(axis_segments))]

    distributions = [exhaustive_distribution_by_r[r] for r in r_values]
    box_handle = None
    for idx, ax in enumerate(axes):
        box = _draw_distribution_boxes_percentile(ax, distribution_positions, distributions, '#6C85A4', label='All combinations (5th-95th)', widths=0.42)
        if box_handle is None:
            box_handle = box['boxes'][0]
        _draw_event_points(ax, dss_positions, dss_eventwise_by_r, '#E07A5F', 'o', label='DSS')
        _draw_event_points(ax, optimum_positions, optimum_eventwise_by_r, '#3D9970', 's', label='Exhaustive optimum')
        ax.set_ylim(*axis_segments[idx])
        if idx == 0:
            ax.set_title(
                f'Exhaustive Benchmark Percentile Boxplot ({title_suffix})' if title_suffix else 'Exhaustive Benchmark Percentile Boxplot',
                fontsize=title_size
            )

    _style_broken_axis_stack(axes, show_xlabel=True)
    axes[-1].set_xticks(distribution_positions)
    axes[-1].set_xticklabels([str(r) for r in r_values])
    top_stack_center = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
    fig.text(0.035, top_stack_center, 'System-level NSE', va='center', rotation='vertical', fontsize=label_size)

    axes[-1].legend(
        [
            box_handle,
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E07A5F', markeredgecolor='#E07A5F', markersize=7),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#3D9970', markeredgecolor='#3D9970', markersize=7),
        ],
        ['All combinations (5th-95th)', 'DSS', 'Exhaustive optimum'],
        fontsize=11,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        ncol=3
    )

    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.18, hspace=0.05)
    plt.savefig(os.path.join(current_dir, output_filename), bbox_inches='tight')
    plt.close(fig)


def plot_exhaustive_eventwise_gap_only(
    dss_eventwise_by_r,
    optimum_eventwise_by_r,
    current_dir,
    output_filename='Exhaustive_gap_eventwise.png',
    title_suffix='',
):
    title_size = 13
    label_size = 12
    tick_size = 11
    r_values = sorted(dss_eventwise_by_r.keys())
    if not r_values:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 7), dpi=300, sharex=True)
    centers = np.arange(1, len(r_values) + 1, dtype=float)
    dss_pos = centers - 0.14
    opt_pos = centers + 0.14

    first_line = True
    for i, r in enumerate(r_values):
        dss_vals = np.asarray(dss_eventwise_by_r[r], dtype=float)
        opt_vals = np.asarray(optimum_eventwise_by_r[r], dtype=float)
        offsets = _event_offsets(len(dss_vals), 0.0, width=0.08)
        for j, (dss_val, opt_val) in enumerate(zip(dss_vals, opt_vals)):
            x_dss = dss_pos[i] + offsets[j]
            x_opt = opt_pos[i] + offsets[j]
            ax_top.plot(
                [x_dss, x_opt],
                [dss_val, opt_val],
                color='#bdbdbd',
                linewidth=0.9,
                zorder=1,
                label='Event pair' if first_line else None,
            )
            first_line = False
            ax_top.scatter(x_dss, dss_val, color='#E07A5F', s=34, marker='o', zorder=3)
            ax_top.scatter(x_opt, opt_val, color='#3D9970', s=34, marker='s', zorder=3)

        ax_top.hlines(np.median(dss_vals), dss_pos[i] - 0.08, dss_pos[i] + 0.08, colors='#E07A5F', linewidth=2.0, zorder=4)
        ax_top.hlines(np.median(opt_vals), opt_pos[i] - 0.08, opt_pos[i] + 0.08, colors='#3D9970', linewidth=2.0, zorder=4)

        gap_vals = opt_vals - dss_vals
        gap_offsets = _event_offsets(len(gap_vals), centers[i], width=0.16)
        ax_bottom.scatter(gap_offsets, gap_vals, color='#7A3E9D', s=32, marker='D', zorder=3)
        ax_bottom.hlines(np.median(gap_vals), centers[i] - 0.1, centers[i] + 0.1, colors='#7A3E9D', linewidth=2.0, zorder=4)

    if title_suffix:
        ax_top.set_title(f'Exhaustive Benchmark System-level Comparison ({title_suffix})', fontsize=title_size)
    else:
        ax_top.set_title('Exhaustive Benchmark System-level Comparison', fontsize=title_size)

    ax_top.set_ylabel('System-level NSE', fontsize=label_size)
    ax_top.tick_params(axis='both', labelsize=tick_size)
    ax_top.spines['right'].set_visible(True)
    ax_top.spines['left'].set_visible(True)
    ax_top.set_ylim(0.80, 1.00)

    ax_bottom.axhline(0.0, color='black', linewidth=1.0, linestyle='--')
    ax_bottom.set_ylabel('NSE Gap', fontsize=label_size)
    ax_bottom.set_xlabel('Number of Sensors (r)', fontsize=label_size)
    ax_bottom.tick_params(axis='both', labelsize=tick_size)
    ax_bottom.spines['right'].set_visible(True)
    ax_bottom.spines['left'].set_visible(True)
    ax_bottom.set_xticks(centers)
    ax_bottom.set_xticklabels([str(r) for r in r_values])

    ax_top.legend(
        [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E07A5F', markeredgecolor='#E07A5F', markersize=7),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#3D9970', markeredgecolor='#3D9970', markersize=7),
            Line2D([0], [0], color='#bdbdbd', linewidth=1.0),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#7A3E9D', markeredgecolor='#7A3E9D', markersize=7),
        ],
        ['DSS', 'Exhaustive optimum', 'Event pair', 'System-level gap'],
        fontsize=11,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
    )

    fig.subplots_adjust(left=0.12, right=0.96, top=0.93, bottom=0.16, hspace=0.18)
    plt.savefig(os.path.join(current_dir, output_filename), bbox_inches='tight')
    plt.close(fig)


def plot_random_monte_carlo_eventwise_benchmark(
    random_distribution_by_r,
    dss_eventwise_by_r,
    current_dir,
    output_filename='Random_benchmark_eventwise.png',
    title_suffix='',
    show_panel_label=False,
):
    title_size = 13
    label_size = 13
    tick_size = 12
    r_values = sorted(random_distribution_by_r.keys())
    if not r_values:
        return

    distribution_positions = np.arange(1, len(r_values) + 1, dtype=float)
    dss_positions = distribution_positions
    axis_segments = _broken_axis_limits(random_distribution_by_r, [dss_eventwise_by_r])
    fig, axes = plt.subplots(len(axis_segments), 1, figsize=(8, 10), dpi=300, sharex=True)
    axes = np.atleast_1d(axes)

    distributions = [random_distribution_by_r[r] for r in r_values]
    box_handle = None
    for idx, ax in enumerate(axes):
        box = _draw_distribution_boxes(ax, distribution_positions, distributions, '#aa2b46', label='Random', widths=0.44)
        if box_handle is None:
            box_handle = box['boxes'][0]
        _draw_event_points(ax, dss_positions, dss_eventwise_by_r, '#f8dfa6', 'o', label='DSS')
        ax.set_ylim(*axis_segments[idx])
        if idx == 0 and show_panel_label:
            ax.text(0.015, 0.96, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

    _style_broken_axis_stack(axes, show_xlabel=True)
    axes[-1].set_xticks(distribution_positions)
    axes[-1].set_xticklabels([str(r) for r in r_values])
    axes[-1].legend(
        [box_handle],
        ['Random'],
        fontsize=8.9,
        loc='center left',
        bbox_to_anchor=(0.83, -0.14),
        frameon=True,
        ncol=1,
        columnspacing=0.6,
        handletextpad=0.4,
        borderpad=0.25
    )

    stack_center = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
    fig.text(0.04, stack_center, 'System-level NSE', va='center', rotation='vertical', fontsize=label_size)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.22, hspace=0.05)
    plt.savefig(os.path.join(current_dir, output_filename), bbox_inches='tight')
    plt.close(fig)


def plot_random_monte_carlo_eventwise_benchmark_percentile(
    random_distribution_by_r,
    dss_eventwise_by_r,
    current_dir,
    output_filename='Random_benchmark_eventwise_percentile.png',
    title_suffix='',
    axis_segments=None,
):
    title_size = 13
    label_size = 12
    r_values = sorted(random_distribution_by_r.keys())
    if not r_values:
        return

    distribution_positions = np.arange(1, len(r_values) + 1, dtype=float)
    dss_positions = distribution_positions
    axis_segments = axis_segments or _broken_axis_limits(random_distribution_by_r, [dss_eventwise_by_r])
    fig, axes = plt.subplots(len(axis_segments), 1, figsize=(8, 8), dpi=300, sharex=True)
    axes = np.atleast_1d(axes)

    distributions = [random_distribution_by_r[r] for r in r_values]
    box_handle = None
    for idx, ax in enumerate(axes):
        box = _draw_distribution_boxes_percentile(ax, distribution_positions, distributions, '#6C85A4', label='Random (5th-95th)')
        if box_handle is None:
            box_handle = box['boxes'][0]
        _draw_event_points(ax, dss_positions, dss_eventwise_by_r, '#EBD8B7', 'o', label='DSS')
        ax.set_ylim(*axis_segments[idx])
        if idx == 0:
            ax.set_title(
                f'Random Benchmark Percentile Boxplot ({title_suffix})' if title_suffix else 'Random Benchmark Percentile Boxplot',
                fontsize=title_size
            )

    _style_broken_axis_stack(axes, show_xlabel=True)
    axes[-1].set_xticks(distribution_positions)
    axes[-1].set_xticklabels([str(r) for r in r_values])
    top_stack_center = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
    fig.text(0.035, top_stack_center, 'System-level NSE', va='center', rotation='vertical', fontsize=label_size)
    axes[-1].legend(
        [
            box_handle,
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#EBD8B7', markeredgecolor='#EBD8B7', markersize=7),
        ],
        ['Random (5th-95th)', 'DSS'],
        fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        ncol=2
    )

    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.18, hspace=0.05)
    plt.savefig(os.path.join(current_dir, output_filename), bbox_inches='tight')
    plt.close(fig)


def plot_psi_heatmap(
    Psi,
    current_dir,
    cmap='viridis',
    cbar_width="5%",
    cbar_pad="3%",
    cbar_position='right',
    title_suffix='',
    output_filename='Psi_heatmap.png'
):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns_heatmap = sns.heatmap(Psi, ax=ax, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    divider = make_axes_locatable(ax)

    if cbar_position == 'right':
        cax = divider.append_axes("right", size=cbar_width, pad=cbar_pad)
    elif cbar_position == 'left':
        cax = divider.append_axes("left", size=cbar_width, pad=cbar_pad)
    elif cbar_position == 'top':
        cax = divider.append_axes("top", size=cbar_width, pad=cbar_pad)
        cax.orientation = 'horizontal'
    elif cbar_position == 'bottom':
        cax = divider.append_axes("bottom", size=cbar_width, pad=cbar_pad)
        cax.orientation = 'horizontal'
    else:
        raise ValueError("cbar_position must be one of 'right', 'left', 'top', 'bottom'")

    cbar = fig.colorbar(sns_heatmap.get_children()[0], cax=cax)
    cbar.set_label('value')
    if title_suffix:
        ax.set_title(f'Psi Heatmap ({title_suffix})')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, output_filename))
    plt.close()


def plot_residuals_heatmap(
    Node_ID,
    all_pivots,
    all_metric_values,
    current_dir,
    title_suffix='',
    output_filename='residuals_heatmap.png',
    colorbar_label='Relative Projection Residual',
    font_settings=None,
    location_map_path=None,
    location_inset_bounds=(0.010, 0.020, 0.30, 0.56),
    cmap='RdYlBu_r',
    vmin=None,
    vmax=None,
    center=None
):
    r_values = range(2, 2 + len(all_metric_values))
    max_sensors = max([len(values) for values in all_metric_values])
    heatmap_data = pd.DataFrame(index=range(max_sensors), columns=r_values)
    node_ids = pd.DataFrame(index=range(max_sensors), columns=r_values)

    for idx, r in enumerate(r_values):
        current_values = all_metric_values[idx]
        current_pivot = all_pivots[idx]
        for i, value in enumerate(current_values):
            try:
                heatmap_data.loc[i, r] = float(value) if value is not None else np.nan
            except (ValueError, TypeError):
                heatmap_data.loc[i, r] = np.nan
        for i, p in enumerate(current_pivot[:len(current_values)]):
            try:
                p_value = p.item() if hasattr(p, 'item') else p[0] if hasattr(p, '__len__') and not isinstance(p, str) else p
                node_ids.loc[i, r] = Node_ID[int(p_value)]
            except (IndexError, TypeError, ValueError):
                node_ids.loc[i, r] = "N/A"

    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    heatmap_data = heatmap_data.mask(np.isinf(heatmap_data), np.nan)
    plt.figure(figsize=(len(r_values) * 1.5, max_sensors * 0.8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap=cmap, linewidths=0,
                     vmin=vmin, vmax=vmax, center=center,
                     cbar_kws={'label': colorbar_label, 'pad': 0.02})
    cbar = ax.collections[0].colorbar
    if font_settings is None:
        label_fontsize = 14
        tick_fontsize = 12
        cbar_fontsize = 14
    else:
        label_fontsize = font_settings.get('label_fontsize', 14)
        tick_fontsize = font_settings.get('tick_fontsize', 12)
        cbar_fontsize = font_settings.get('cbar_fontsize', 14)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    if font_settings is None:
        font_choice = input("Change font sizes? (y/n, default: n): ").strip().lower()
    else:
        font_choice = 'n'
    if font_settings is None and font_choice == 'y':
        try:
            label_fontsize = int(input("Enter axis label font size (default: 14): ") or "14")
            tick_fontsize = int(input("Enter tick font size (default: 12): ") or "12")
            cbar_fontsize = int(input("Enter colorbar label font size (default: 14): ") or "14")
        except ValueError:
            print("Invalid input. Using default font sizes.")
            label_fontsize = 14
            tick_fontsize = 12
            cbar_fontsize = 14
    cbar.ax.set_ylabel(colorbar_label, fontsize=cbar_fontsize)
    ax.set_xlabel('r (number of sensor used)', fontsize=label_fontsize)
    ax.set_ylabel('Sensor Rank', fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    xticks_pos = np.arange(len(r_values)) + 0.5
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(r_values)
    yticks_pos = np.arange(max_sensors) + 0.5
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(range(1, max_sensors + 1))

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    for i in range(max_sensors):
        for j, r in enumerate(r_values):
            if not pd.isna(heatmap_data.loc[i, r]):
                node_id = node_ids.loc[i, r]
                if node_id is not None and not pd.isna(node_id):
                    ax.text(j + 0.5, i + 0.5, f"{node_id}", horizontalalignment='center',
                            verticalalignment='center', color='black', weight='bold', fontsize=14)

    if location_map_path is None:
        location_map_path = os.path.join(current_dir, 'Sensor_full.png')
    if location_map_path and os.path.exists(location_map_path):
        try:
            location_img = Image.open(location_map_path)
            bbox = location_img.getbbox()
            if bbox is not None:
                location_img = location_img.crop(bbox)
            inset_ax = ax.inset_axes(location_inset_bounds, transform=ax.transAxes, zorder=10)
            inset_ax.imshow(location_img)
            inset_ax.set_axis_off()
            inset_ax.set_facecolor('none')
        except Exception as exc:
            print(f"Warning: failed to add sensor-location inset map: {exc}")

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, output_filename), dpi=300, bbox_inches='tight')
    plt.close()
    return {
        'label_fontsize': label_fontsize,
        'tick_fontsize': tick_fontsize,
        'cbar_fontsize': cbar_fontsize,
    }


def plot_hydrograph_with_rainfall(
    flow_csv_path,
    rainfall_excel_path,
    current_dir,
    selected_nodes=None,
    rainfall_column=0,
    flow_ylim=None,
    rainfall_ylim=None,
    output_filename='rainfall_flowrate.png'
):
    def _build_datetime_axis(labels):
        base_date = datetime.datetime(2000, 1, 1)
        axis = []
        day_offset = 0
        previous_time = None
        for label in labels:
            parsed_time = datetime.datetime.strptime(str(label), '%H:%M:%S').time()
            if previous_time is not None and parsed_time < previous_time:
                day_offset += 1
            axis.append(datetime.datetime.combine(base_date.date(), parsed_time) + datetime.timedelta(days=day_offset))
            previous_time = parsed_time
        return axis

    flow_data = pd.read_csv(flow_csv_path)
    flow_columns = flow_data.columns[1:]
    time_points = []
    for col in flow_columns:
        if ' at ' in str(col):
            time_points.append(str(col).split(' at ', 1)[1])
        else:
            time_points.append(str(col))
    flow_time_axis = _build_datetime_axis(time_points)

    if selected_nodes is None:
        selected_nodes = flow_data['Node ID'].tolist()
    selected_flow_data = flow_data[flow_data['Node ID'].isin(selected_nodes)].copy()

    fig, ax1 = plt.subplots(figsize=(15, 8))
    for _, row in selected_flow_data.iterrows():
        node_id = row['Node ID']
        flow_values = row.iloc[1:].to_numpy(dtype=float)
        ax1.plot(flow_time_axis, flow_values, label=f'Node {node_id}', linewidth=2)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Flow Rate (CFS)')
    ax1.tick_params(axis='y')
    ax1.legend(loc='center', bbox_to_anchor=(0.92, 0.5), framealpha=0.7, fontsize=14)
    ax1.grid(True, alpha=0.3)
    if flow_ylim is not None:
        ax1.set_ylim(flow_ylim[0], flow_ylim[1])
    else:
        ax1.set_ylim(0, 6)

    try:
        rainfall_df = pd.read_excel(rainfall_excel_path)
        if rainfall_df.shape[1] > rainfall_column:
            rainfall_values = rainfall_df.iloc[:, rainfall_column].to_numpy(dtype=float)
        else:
            rainfall_values = rainfall_df.iloc[:, 0].to_numpy(dtype=float)
        if rainfall_df.shape[1] > 1:
            rainfall_time_labels = [
                item.strftime('%H:%M:%S') if hasattr(item, 'strftime') else str(item)
                for item in rainfall_df.iloc[:, 1].tolist()
            ]
        else:
            rainfall_time_labels = time_points[:len(rainfall_values)]
    except Exception as e:
        print(f"Error reading rainfall data: {e}")
        rainfall_values = np.zeros(len(time_points))
        rainfall_time_labels = time_points[:len(rainfall_values)]

    rainfall_time_axis = _build_datetime_axis(rainfall_time_labels)

    if len(rainfall_values) != len(rainfall_time_axis):
        aligned_length = min(len(rainfall_values), len(rainfall_time_axis))
        rainfall_values = rainfall_values[:aligned_length]
        rainfall_time_axis = rainfall_time_axis[:aligned_length]

    ax2 = ax1.twinx()
    bar_width = datetime.timedelta(minutes=4) / datetime.timedelta(days=1)
    ax2.bar(rainfall_time_axis, rainfall_values, color='#15559a', width=bar_width, label='Rainfall')
    ax2.set_ylabel('Rainfall Volume (in)')
    ax2.invert_yaxis()
    if rainfall_ylim is not None:
        ax2.set_ylim(rainfall_ylim[1], rainfall_ylim[0])
    else:
        ax2.set_ylim(0.06, 0)

    tick_times = [
        datetime.datetime(2000, 1, 1, 0, 0),
        datetime.datetime(2000, 1, 1, 2, 0),
        datetime.datetime(2000, 1, 1, 4, 0),
        datetime.datetime(2000, 1, 1, 6, 0),
        datetime.datetime(2000, 1, 1, 8, 0),
        datetime.datetime(2000, 1, 1, 10, 0),
        datetime.datetime(2000, 1, 1, 12, 0),
        datetime.datetime(2000, 1, 1, 14, 0),
        datetime.datetime(2000, 1, 1, 16, 0),
        datetime.datetime(2000, 1, 1, 18, 0),
        datetime.datetime(2000, 1, 1, 20, 0),
        datetime.datetime(2000, 1, 1, 22, 0),
        datetime.datetime(2000, 1, 2, 0, 0),
    ]
    tick_labels = [
        '00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00',
        '12:00:00', '14:00:00', '16:00:00', '18:00:00', '20:00:00', '22:00:00', '24:00:00'
    ]
    ax1.set_xlim(datetime.datetime(2000, 1, 1, 0, 0), datetime.datetime(2000, 1, 2, 0, 0))
    ax1.set_xticks(tick_times)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            x_fontsize = int(input("Enter x-axis font size (default: 16): ") or "16")
            y_fontsize = int(input("Enter y-axis font size (default: 16): ") or "16")
        except ValueError:
            print("Invalid input. Using default font size 14.")
            x_fontsize = 16
            y_fontsize = 16
        ax1.set_xlabel('Time', fontsize=x_fontsize)
        ax1.set_ylabel('Flow Rate (CFS)', fontsize=y_fontsize)
        ax2.set_ylabel('Rainfall Volume (in)', fontsize=y_fontsize)
        tick_choice = input("Also change tick font size? (y/n, default: n): ").strip().lower()
        if tick_choice == 'y':
            try:
                x_tick = int(input("Enter x-axis tick font size (default: 14): ") or "14")
                y_tick = int(input("Enter y-axis tick font size (default: 14): ") or "14")
                ax1.tick_params(axis='y', labelsize=y_tick)
                ax2.tick_params(axis='y', labelsize=y_tick)
                ax1.tick_params(axis='x', labelsize=x_tick)
                ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=x_tick)
            except ValueError:
                print("Invalid input. Using default tick font size.")
    else:
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Flow Rate (CFS)')
        ax2.set_ylabel('Rainfall Volume (in)')

    plt.tight_layout()
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hydrograph saved to: {output_path}")


def plot_mean_residuals_NSE(residuals_nse_mean, current_dir, ylabel='Mean Delta NSE',
                            output_filename='residuals_delta_nse.png'):
    node_ids = np.array(list(residuals_nse_mean.keys()))
    x_values = np.array([residuals_nse_mean[x][1] for x in node_ids])
    y_values = np.array([residuals_nse_mean[x][0] for x in node_ids])
    x_errors = np.array([residuals_nse_mean[x][3] for x in node_ids])
    y_errors = np.array([residuals_nse_mean[x][2] for x in node_ids])
    unique_node_ids = list(set(node_ids))
    colors = plt.cm.tab20(range(len(unique_node_ids)))
    node_to_color = dict(zip(unique_node_ids, colors))
    scatter_colors = [node_to_color[node] for node in node_ids]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.scatter(x_values, y_values, c=scatter_colors, label='Data Points', s=48)
    plt.errorbar(x_values, y_values, xerr=x_errors, yerr=y_errors, fmt='none', ecolor='black', alpha=0.5, capsize=3)

    custom_positions = {
        '137': (-15, -20, 'center'),
        'J304': (-15, -20, 'center'),
        'J305': (-15, 5, 'center'),
        '96': (-15, -20, 'center'),
        '184': (5, 2, 'left')
    }
    for i, node_id in enumerate(node_ids):
        offset_x, offset_y, ha_align = custom_positions.get(node_id, (5, 5, 'left'))
        plt.annotate(node_id, (x_values[i], y_values[i]), xytext=(offset_x, offset_y),
                     textcoords='offset points', fontsize=14, ha=ha_align, va='bottom')

    x_reshaped = x_values.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(x_reshaped, y_values)
    y_pred = linear_model.predict(x_reshaped)
    r2 = r2_score(y_values, y_pred)
    slope = linear_model.coef_[0]
    intercept = linear_model.intercept_
    x_smooth = np.linspace(x_values.min(), x_values.max(), 100)
    y_smooth = linear_model.predict(x_smooth.reshape(-1, 1))
    plt.plot(x_smooth, y_smooth, color='red', linewidth=2,
             label=f'Linear Fit: y = {slope:.4f}x + {intercept:.4f}\nR2 = {r2:.4f}')
    plt.text(0.02, 0.125, f'y = {slope:.4f}x + {intercept:.4f}\nR2 = {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=16, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    font_choice = input("Change axis font size? (y/n, default: n): ").strip().lower()
    if font_choice == 'y':
        try:
            x_fontsize = int(input("Enter x-axis font size (default: 14): ") or "14")
            y_fontsize = int(input("Enter y-axis font size (default: 14): ") or "14")
        except ValueError:
            print("Invalid input. Using default font size 12.")
            x_fontsize = 14
            y_fontsize = 14
        ax.set_xlabel('Mean Relative Projection Residual', fontsize=x_fontsize)
        ax.set_ylabel(ylabel, fontsize=y_fontsize)
        tick_choice = input("Also change tick font size? (y/n, default: n): ").strip().lower()
        if tick_choice == 'y':
            x_tick = int(input("Enter x-axis tick font size (default: 12): ") or "12")
            y_tick = int(input("Enter y-axis tick font size (default: 12): ") or "12")
            ax.tick_params(axis='x', labelsize=x_tick)
            ax.tick_params(axis='y', labelsize=y_tick)
    else:
        ax.set_xlabel('Mean Relative Projection Residual')
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, output_filename), dpi=300)
    plt.close()


def plot_dropout_diagnostic_scatter(df, x_col, y_col, current_dir, output_filename, xlabel, ylabel='Delta Mean NSE'):
    x_values = df[x_col].to_numpy(dtype=float)
    y_values = df[y_col].to_numpy(dtype=float)
    node_ids = df['Node_ID'].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x_values, y_values, s=56, color='#4a98c5')

    for i, node_id in enumerate(node_ids):
        ax.annotate(node_id, (x_values[i], y_values[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=12, ha='left', va='bottom')

    if len(x_values) >= 2 and np.std(x_values) > 0 and np.std(y_values) > 0:
        linear_model = LinearRegression()
        linear_model.fit(x_values.reshape(-1, 1), y_values)
        y_pred = linear_model.predict(x_values.reshape(-1, 1))
        r2 = r2_score(y_values, y_pred)
        slope = linear_model.coef_[0]
        intercept = linear_model.intercept_
        x_smooth = np.linspace(x_values.min(), x_values.max(), 100)
        y_smooth = linear_model.predict(x_smooth.reshape(-1, 1))
        ax.plot(x_smooth, y_smooth, color='red', linewidth=2)
        ax.text(0.02, 0.95, f'y = {slope:.4f}x + {intercept:.4f}\nR2 = {r2:.4f}',
                transform=ax.transAxes, fontsize=14, va='top',
                bbox=dict(facecolor='white', alpha=0.85))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, output_filename), dpi=300)
    plt.close()


def plot_dropout_regime_panels(
    diagnostics_by_r,
    current_dir,
    x_col='Relative_Projection_Residual',
    xlabel='Relative Projection Residual',
    output_filename='dropout_regime_transition.png',
    y_col='Delta_Mean_NSE',
    ylabel='Delta Mean NSE'
):
    r_values = list(diagnostics_by_r.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    axes = axes.flatten()
    panel_tags = ['(a)', '(b)', '(c)', '(d)']
    label_offsets = {
        4: {'J122': (-12, 10, 'right', 'bottom'), '182': (8, -10, 'left', 'top')},
        6: {
            'J306': (10, -2, 'left', 'center'),
            'J122': (-14, 10, 'right', 'bottom'),
            '182': (8, -10, 'left', 'top'),
            '154': (8, 8, 'left', 'bottom'),
            '20': (6, 8, 'left', 'bottom')
        },
        8: {
            'J63': (-8, -10, 'right', 'top'),
            '182': (-8, -10, 'right', 'top'),
            'J62': (8, 8, 'left', 'bottom'),
            '152': (8, 6, 'left', 'bottom'),
            '185': (8, 2, 'left', 'center'),
            'J305': (-8, 2, 'right', 'center'),
            'OF-03': (8, 2, 'left', 'center')
        },
        10: {
            'J122': (0, -12, 'center', 'top'),
            'J126': (0, -12, 'center', 'top'),
            '182': (0, -12, 'center', 'top'),
            'J62': (4, 8, 'left', 'bottom'),
            '152': (6, 8, 'left', 'bottom'),
            '20': (6, 2, 'left', 'center'),
            'J305': (6, 4, 'left', 'bottom'),
            '179': (-6, 4, 'right', 'bottom'),
            'OF-03': (6, 4, 'left', 'bottom'),
            'J63': (6, 4, 'left', 'bottom')
        },
    }

    for idx, r in enumerate(r_values):
        ax = axes[idx]
        df = diagnostics_by_r[r].copy()
        x_values = df[x_col].to_numpy(dtype=float)
        y_values = df[y_col].to_numpy(dtype=float)
        node_ids = df['Node_ID'].astype(str).to_numpy()

        ax.scatter(x_values, y_values, s=56, color='#4a98c5', edgecolors='white', linewidths=0.6)

        x_pad = (x_values.max() - x_values.min()) * 0.12 if len(x_values) > 1 else 0.1
        y_pad = (y_values.max() - y_values.min()) * 0.12 if len(y_values) > 1 else 0.1
        x_min = x_values.min() - x_pad
        x_max = x_values.max() + x_pad
        y_min = min(0.0, y_values.min() - y_pad)
        y_max = y_values.max() + y_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for i, node_id in enumerate(node_ids):
            offset_x, offset_y, ha_align, va_align = label_offsets.get(r, {}).get(
                node_id, (5, 5, 'left', 'bottom')
            )
            ax.annotate(
                node_id,
                (x_values[i], y_values[i]),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=9,
                ha=ha_align,
                va=va_align,
                clip_on=True
            )

        if len(df) >= 2 and np.std(x_values) > 0 and np.std(y_values) > 0:
            linear_model = LinearRegression()
            linear_model.fit(x_values.reshape(-1, 1), y_values)
            y_pred = linear_model.predict(x_values.reshape(-1, 1))
            r2 = r2_score(y_values, y_pred)
            slope = linear_model.coef_[0]
            intercept = linear_model.intercept_
            x_smooth = np.linspace(x_values.min(), x_values.max(), 100)
            y_smooth = linear_model.predict(x_smooth.reshape(-1, 1))
            ax.plot(x_smooth, y_smooth, color='#d65244', linewidth=2)
            fit_box = AnchoredText(
                f'r = {r}\ny = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r2:.3f}',
                loc='upper right',
                prop=dict(size=9.5),
                frameon=True,
                borderpad=0.65,
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=ax.transAxes
            )
            fit_box.patch.set_facecolor('white')
            fit_box.patch.set_alpha(0.9)
            fit_box.patch.set_edgecolor('#999999')
            fit_box.patch.set_boxstyle('round,pad=0.35')
            fit_box.txt._text.set_multialignment('left')
            fit_box.txt._text.set_ha('left')
            ax.add_artist(fit_box)
        else:
            fit_box = AnchoredText(
                f'r = {r}\nInsufficient variance',
                loc='upper right',
                prop=dict(size=9.5),
                frameon=True,
                borderpad=0.65,
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=ax.transAxes
            )
            fit_box.patch.set_facecolor('white')
            fit_box.patch.set_alpha(0.9)
            fit_box.patch.set_edgecolor('#999999')
            fit_box.patch.set_boxstyle('round,pad=0.35')
            fit_box.txt._text.set_multialignment('left')
            fit_box.txt._text.set_ha('left')
            ax.add_artist(fit_box)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.16, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor('#4a4a4a')
        ax.text(0.02, 0.02, panel_tags[idx], transform=ax.transAxes, ha='left', va='bottom',
                fontsize=12, fontweight='bold')

    for idx in range(len(r_values), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, output_filename), dpi=300)
    plt.close()


def plot_dropout_r10_case_panels(summary_df, details_df, current_dir, output_filename='dropout_r10_case_panels.png'):
    selected = summary_df.sort_values('rank').copy()
    fig, axes = plt.subplots(1, len(selected), figsize=(16, 5.5), dpi=300, sharey=False)
    if len(selected) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, selected.iterrows()):
        sensor_id = row['drop_sensor']
        sensor_details = details_df[details_df['drop_sensor'] == sensor_id].copy()
        sensor_details = sensor_details.sort_values('node_delta_nse', ascending=True)

        labels = [
            f"{node}\nEu={eu:.0f}, Net={'NA' if pd.isna(net) else f'{net:.0f}'}"
            for node, eu, net in zip(
                sensor_details['affected_node'],
                sensor_details['euclidean_distance'],
                sensor_details['network_distance']
            )
        ]
        colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(sensor_details)))
        ax.barh(labels, sensor_details['node_delta_nse'], color=colors, edgecolor='white')
        ax.set_xlabel('Dropout-induced Reduction in Node-level NSE')
        ax.set_title(
            f"Drop {sensor_id}\nMean Delta NSE = {row['delta_mean_nse']:.3f}\nConnected Top5 = {int(row['connected_count_top5'])}/5",
            fontsize=11
        )
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, output_filename), dpi=300)
    plt.close()


def _dropout_base_paths(current_dir=None, normalization_mode='global_minmax'):
    base_dir = current_dir or os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, f'Dropout_Discussion_{normalization_mode}')
    combined_csv = os.path.join(base_dir, f'dropout_system_level_diagnostics_{normalization_mode}_r2_r10.csv')
    old_level_name = 'system' + 'wise'
    legacy_combined_csv = os.path.join(base_dir, f'dropout_{old_level_name}_diagnostics_{normalization_mode}_r2_r10.csv')
    if not os.path.exists(combined_csv) and os.path.exists(legacy_combined_csv):
        combined_csv = legacy_combined_csv
    return base_dir, output_dir, combined_csv


def _read_dropout_diagnostics_csv(path):
    df = pd.read_csv(path)
    old_level_name = 'System' + 'wise'
    legacy_columns = {
        'Baseline_' + old_level_name + '_Mean_NSE': 'Baseline_System_Level_Mean_NSE',
        'Dropout_' + old_level_name + '_Mean_NSE': 'Dropout_System_Level_Mean_NSE',
        'Delta_' + old_level_name + '_Mean_NSE': 'Delta_System_Level_Mean_NSE',
    }
    return df.rename(columns={old: new for old, new in legacy_columns.items() if old in df.columns})


def _dropout_within_r_zscore(df, columns):
    out = df.copy()
    for col in columns:
        def _zscore_group(s):
            std = s.std(ddof=0)
            if np.isfinite(std) and std > 0:
                return (s - s.mean()) / std
            return 0.0

        out[f'{col}_z'] = out.groupby('r')[col].transform(_zscore_group)
    return out


def _parse_dropout_event_list(text):
    if isinstance(text, list):
        return [float(v) for v in text]
    return [float(v) for v in ast.literal_eval(text)]


def _row_value_with_legacy(row, preferred_name, legacy_name):
    if preferred_name in row:
        return row[preferred_name]
    return row[legacy_name]


def _load_dropout_qr_order(psi, r_values):
    qr_orders = {}
    for r in r_values:
        pivots = qr(psi[:, :r].T, pivoting=True)[2][:r]
        qr_orders[r] = {int(idx): order + 1 for order, idx in enumerate(pivots)}
    return qr_orders


def _dropout_eventwise_delta_values(row):
    old_level_name = 'System' + 'wise'
    baseline_values = np.asarray(
        _parse_dropout_event_list(_row_value_with_legacy(row, 'Baseline_System_Level_NSE', 'Baseline_Eventwise_' + old_level_name + '_NSE')),
        dtype=float
    )
    dropout_values = np.asarray(
        _parse_dropout_event_list(_row_value_with_legacy(row, 'Dropout_System_Level_NSE', 'Dropout_Eventwise_' + old_level_name + '_NSE')),
        dtype=float
    )
    n = min(baseline_values.size, dropout_values.size)
    if n == 0:
        return []
    return (baseline_values[:n] - dropout_values[:n]).tolist()


def _draw_dropout_sensor_panel(sensor_ax, sensor_image_path):
    sensor_ax.set_xticks([])
    sensor_ax.set_yticks([])
    for spine in sensor_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')
    if os.path.exists(sensor_image_path):
        sensor_image = plt.imread(sensor_image_path)
        if sensor_image.ndim == 3:
            rgb = sensor_image[:, :, :3]
            if sensor_image.shape[2] >= 4:
                alpha = sensor_image[:, :, 3]
                mask = (alpha > 0) & np.any(rgb < 0.98, axis=2)
            else:
                mask = np.any(rgb < 0.98, axis=2)
            ys, xs = np.where(mask)
            if xs.size and ys.size:
                pad = 8
                x0 = max(int(xs.min()) - pad, 0)
                x1 = min(int(xs.max()) + pad + 1, sensor_image.shape[1])
                y0 = max(int(ys.min()) - pad, 0)
                y1 = min(int(ys.max()) + pad + 1, sensor_image.shape[0])
                sensor_image = sensor_image[y0:y1, x0:x1]
        sensor_ax.imshow(sensor_image)
        sensor_ax.set_aspect('equal', adjustable='box')
        sensor_ax.set_anchor('W')
    else:
        sensor_ax.text(
            0.5,
            0.5,
            f'{os.path.basename(sensor_image_path)}\nnot found',
            ha='center',
            va='center',
            fontsize=11,
            color='#555555',
            transform=sensor_ax.transAxes,
        )
    sensor_ax.set_xlim(sensor_ax.get_xlim())
    sensor_ax.set_ylim(sensor_ax.get_ylim())


def generate_dropout_combined_delta_summary(psi, r_values=None, current_dir=None, normalization_mode='global_minmax'):
    _, output_dir, combined_csv = _dropout_base_paths(current_dir=current_dir, normalization_mode=normalization_mode)
    os.makedirs(output_dir, exist_ok=True)
    if r_values is None:
        r_values = [6, 10]
    r_values = [int(r) for r in r_values]
    qr_orders = _load_dropout_qr_order(psi, r_values)
    summary_df = _read_dropout_diagnostics_csv(combined_csv)

    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(10.2, 8.2), dpi=300)
    height_ratios = []
    for row_index in range(len(r_values)):
        height_ratios.append(1.0)
        height_ratios.append(0.08 if row_index < len(r_values) - 1 else 0.16)
    height_ratios.append(0.82)
    gs = fig.add_gridspec(
        len(height_ratios),
        2,
        width_ratios=[2.45, 0.78],
        height_ratios=height_ratios,
        hspace=0.04,
        wspace=0.00,
    )

    box_axes = []
    sensor_axes = []
    shared_axis = None
    top_delta_values = []
    for row_index, r in enumerate(r_values):
        grid_row = row_index * 2
        ax_box = fig.add_subplot(gs[grid_row, 0], sharey=shared_axis)
        if shared_axis is None:
            shared_axis = ax_box
        ax_sensor = fig.add_subplot(gs[grid_row, 1])
        box_axes.append(ax_box)
        sensor_axes.append(ax_sensor)

        path = os.path.join(output_dir, f'dropout_system_level_diagnostics_r{r}.csv')
        legacy_path = os.path.join(output_dir, f'dropout_{"system" + "wise"}_diagnostics_r{r}.csv')
        if not os.path.exists(path) and os.path.exists(legacy_path):
            path = legacy_path
        df = _read_dropout_diagnostics_csv(path)
        df['pivot_order'] = df['Sensor_Index'].map(qr_orders[r])
        df = df.sort_values('pivot_order')

        box_data = [_dropout_eventwise_delta_values(row) for _, row in df.iterrows()]
        top_delta_values.extend(
            value for values in box_data for value in values if np.isfinite(value)
        )
        labels = df['Node_ID'].astype(str).tolist()
        positions = np.arange(1, len(box_data) + 1)
        ax_box.boxplot(
            box_data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            boxprops=dict(facecolor='#8fb1cc', edgecolor='#4a4a4a', linewidth=1.5),
            medianprops=dict(color='#222222', linewidth=1.5),
            whiskerprops=dict(color='#4a4a4a', linewidth=1.4),
            capprops=dict(color='#4a4a4a', linewidth=1.4),
            flierprops=dict(marker='o', markersize=2.6, markerfacecolor='#4a98c5', markeredgecolor='none', alpha=0.45)
        )
        ax_box.axhline(0.0, color='#d65244', linestyle='--', linewidth=1.5, label='No-loss reference')
        ax_box.set_xlim(0.35, len(box_data) + 0.65)
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
        ax_box.tick_params(axis='y', labelsize=10, pad=2)
        ax_box.grid(True, axis='y', alpha=0.18)
        ax_box.text(0.01, 0.96, f'{chr(ord("a") + row_index)}  r = {r}', transform=ax_box.transAxes,
                    ha='left', va='top', fontsize=12, fontweight='bold')
        if row_index == 0:
            ax_box.legend(loc='upper right', frameon=False, fontsize=10, facecolor='none')
        for spine in ax_box.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor('#4a4a4a')

        sensor_image_path = os.path.join(current_dir or output_dir, f'Sensor_{r}.png')
        _draw_dropout_sensor_panel(ax_sensor, sensor_image_path)

    if top_delta_values:
        y_min = min(0.0, min(top_delta_values) - 0.03)
        y_max = max(top_delta_values) + 0.06
        for ax in box_axes:
            ax.set_ylim(y_min, y_max)

    box_axes[-1].set_xlabel('Dropped sensor (QR selection order)', fontsize=12, labelpad=2)

    ax_summary = fig.add_subplot(gs[len(r_values) * 2, :])
    summary = summary_df.groupby('r').agg(
        mean_delta=('Delta_System_Level_Mean_NSE', 'mean'),
        median_delta=('Delta_System_Level_Mean_NSE', 'median'),
        max_delta=('Delta_System_Level_Mean_NSE', 'max'),
    ).reset_index()
    ax_summary.plot(summary['r'], summary['mean_delta'], marker='o', linewidth=2, color='#2a7ab0', label='Mean loss')
    ax_summary.plot(summary['r'], summary['median_delta'], marker='s', linewidth=2, color='#58a55c', label='Median loss')
    ax_summary.plot(summary['r'], summary['max_delta'], marker='^', linewidth=2, color='#d65244', label='Worst-case loss')
    ax_summary.set_xlabel('r (Number of sensors used)', fontsize=12)
    ax_summary.set_ylim(0.0, 0.8)
    ax_summary.set_xticks(sorted(summary_df['r'].unique()))
    ax_summary.tick_params(axis='both', labelsize=10, pad=2)
    ax_summary.grid(True, alpha=0.18)
    ax_summary.legend(loc='upper right', frameon=False, fontsize=10, facecolor='none')
    ax_summary.text(0.01, 0.96, f'{chr(ord("a") + len(r_values))}', transform=ax_summary.transAxes,
                    ha='left', va='top', fontsize=12, fontweight='bold')
    for spine in ax_summary.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    fig.subplots_adjust(left=0.090, right=0.985, top=0.975, bottom=0.080)
    fig.canvas.draw()
    left_edge = box_axes[0].get_position().x0
    right_edge = sensor_axes[0].get_position().x1
    summary_pos = ax_summary.get_position()
    ax_summary.set_position([left_edge, summary_pos.y0, right_edge - left_edge, summary_pos.height])
    fig.text(0.035, 0.56, 'Delta system-level NSE', va='center', rotation='vertical', fontsize=12)
    output_path = os.path.join(
        output_dir,
        'dropout_delta_boxplot_summary_r' + '_r'.join(str(r) for r in r_values) + '.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.10)
    plt.close()
    return output_path


def _load_dropout_multifactor_augmented_data(psi, current_dir=None, normalization_mode='global_minmax'):
    _, _, combined_csv = _dropout_base_paths(current_dir=current_dir, normalization_mode=normalization_mode)
    df = _read_dropout_diagnostics_csv(combined_csv)

    extra_rows = []
    for r in range(2, 11):
        pivots = qr(psi[:, :r].T, pivoting=True)[2][:r]
        for order, idx in enumerate(pivots, start=1):
            row = psi[idx, :r]
            extra_rows.append({
                'r': r,
                'Sensor_Index': int(idx),
                'pivot_order': order,
                'row_norm_sq': float(np.sum(row ** 2)),
                'row_norm': float(np.linalg.norm(row)),
                'max_abs_loading': float(np.max(np.abs(row))),
            })

    extra_df = pd.DataFrame(extra_rows)
    df = df.merge(extra_df, on=['r', 'Sensor_Index'], how='left')
    df['retained_fraction'] = df['Dropout_System_Level_Mean_NSE'] / df['Baseline_System_Level_Mean_NSE']
    return df


def _fit_dropout_multifactor_standardized_model(df):
    z = _dropout_within_r_zscore(
        df,
        [
            'Delta_System_Level_Mean_NSE',
            'Relative_Projection_Residual',
            'Condition_Number',
            'max_abs_loading',
            'pivot_order',
        ]
    ).dropna()

    x_cols = [
        'Relative_Projection_Residual_z',
        'Condition_Number_z',
        'max_abs_loading_z',
        'pivot_order_z',
    ]
    X = np.column_stack([np.ones(len(z))] + [z[col].to_numpy(dtype=float) for col in x_cols])
    y = z['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    z['pred_z'] = pred
    return z, x_cols, coef, r2


def _fit_dropout_reduced_standardized_model(z):
    x_cols_reduced = [
        'Relative_Projection_Residual_z',
        'Condition_Number_z',
    ]
    X = np.column_stack([np.ones(len(z))] + [z[col].to_numpy(dtype=float) for col in x_cols_reduced])
    y = z['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    z = z.copy()
    z['pred_reduced_z'] = pred
    return z, x_cols_reduced, coef, r2


def _fit_dropout_pivot_only_standardized_model(z):
    x_cols_pivot = ['pivot_order_z']
    X = np.column_stack([np.ones(len(z))] + [z[col].to_numpy(dtype=float) for col in x_cols_pivot])
    y = z['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    z = z.copy()
    z['pred_pivot_z'] = pred
    return z, x_cols_pivot, coef, r2


def _fit_dropout_named_standardized_model(z, x_cols, pred_col):
    X = np.column_stack([np.ones(len(z))] + [z[col].to_numpy(dtype=float) for col in x_cols])
    y = z['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    z = z.copy()
    z[pred_col] = pred
    return z, x_cols, coef, r2


def _add_dropout_residual_inset(ax, z, reduced_color='#4a98c5', full_color='#d65244'):
    residual_reduced = z['Delta_System_Level_Mean_NSE_z'] - z['pred_reduced_z']
    residual_full = z['Delta_System_Level_Mean_NSE_z'] - z['pred_z']
    limit = float(np.nanpercentile(np.abs(np.concatenate([residual_reduced, residual_full])), 98))
    limit = max(limit, 0.25)

    inset = ax.inset_axes([0.035, 0.63, 0.27, 0.21])
    bp = inset.boxplot(
        [residual_reduced, residual_full],
        positions=[1, 2],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='white', linewidth=1.4),
        whiskerprops=dict(color='#666666', linewidth=1.0),
        capprops=dict(color='#666666', linewidth=1.0),
        boxprops=dict(edgecolor='#555555', linewidth=1.0)
    )
    for patch, color in zip(bp['boxes'], [reduced_color, full_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    inset.axhline(0.0, color='#666666', linewidth=0.9, linestyle='--')
    inset.set_ylim(-limit, limit)
    inset.set_xticks([1, 2])
    inset.set_xticklabels(['2-var', 'Full'], fontsize=7.3)
    inset.tick_params(axis='y', labelsize=7.0, length=2)
    inset.set_title('Prediction residuals', fontsize=7.5, pad=1.5)
    inset.grid(True, axis='y', alpha=0.12)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('#666666')


def _load_bootstrap_delta_r2_summary(output_dir):
    summary_path = os.path.join(output_dir, 'dropout_bootstrap_delta_r2_samples.csv')
    feature_labels = {
        'Relative_Projection_Residual': 'RPR',
        'Condition_Number': 'Condition',
        'max_abs_loading': 'Max loading',
        'pivot_order': 'QR rank',
    }
    if not os.path.exists(summary_path):
        return None

    boot = pd.read_csv(summary_path)
    rows = []
    for feature, label in feature_labels.items():
        if feature not in boot:
            continue
        values = boot[feature].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append({
            'feature': label,
            'median': float(np.nanmedian(values)),
            'ci_5': float(np.nanpercentile(values, 5)),
            'ci_95': float(np.nanpercentile(values, 95)),
        })
    if not rows:
        return None
    summary = pd.DataFrame(rows)
    total = float(summary['median'].sum())
    summary['relative_contribution'] = summary['median'] / total if total > 0 else np.nan
    return summary.sort_values('median', ascending=True)


def _load_dropout_multifactor_palette(use_backup=False):
    formal_palette = {
        'model_colors': {
            'Pivot-only model': '#91abd2',
            'Pivot + RPR model': '#f59694',
            'Pivot + RPR + K model': '#bca6cd',
            'Full model': '#fbd3a2',
        },
        'bar_palette_bottom_to_top': [
            '#fbd3a2',
            '#bca6cd',
            '#f59694',
            '#91abd2',
        ],
        'nested_model_colors': [
            '#91abd2',
            '#f59694',
            '#bca6cd',
            '#fbd3a2',
        ],
    }
    if not use_backup:
        return formal_palette

    backup_path = os.path.join(
        os.path.dirname(__file__),
        'palette_backups',
        'dropout_multifactor_palette_backup.py',
    )
    if not os.path.exists(backup_path):
        return formal_palette

    try:
        backup = runpy.run_path(backup_path)
    except Exception:
        return formal_palette

    return {
        'model_colors': backup.get(
            'DROPOUT_MULTIFACTOR_ORIGINAL_MODEL_COLORS',
            formal_palette['model_colors'],
        ),
        'bar_palette_bottom_to_top': backup.get(
            'DROPOUT_MULTIFACTOR_ORIGINAL_BAR_PALETTE_BOTTOM_TO_TOP',
            formal_palette['bar_palette_bottom_to_top'],
        ),
        'nested_model_colors': backup.get(
            'DROPOUT_MULTIFACTOR_ORIGINAL_NESTED_MODEL_COLORS',
            formal_palette['nested_model_colors'],
        ),
    }


def _dropout_r2_for_columns(z, x_cols, row_indices=None):
    cols = ['Delta_System_Level_Mean_NSE_z'] + list(x_cols)
    data = z.iloc[row_indices][cols] if row_indices is not None else z[cols]
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n <= len(x_cols) + 1:
        return np.nan, n

    y = data['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    if np.nanstd(y, ddof=0) <= 0:
        return np.nan, n

    X = np.column_stack([np.ones(n)] + [data[col].to_numpy(dtype=float) for col in x_cols])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan), n


def _dropout_leave_one_variable_out_delta_r2(z, x_cols, row_indices=None):
    full_r2, n_used = _dropout_r2_for_columns(z, x_cols, row_indices=row_indices)
    deltas = {}
    if not np.isfinite(full_r2):
        return full_r2, n_used, {col: np.nan for col in x_cols}

    for col in x_cols:
        reduced_cols = [candidate for candidate in x_cols if candidate != col]
        reduced_r2, _ = _dropout_r2_for_columns(z, reduced_cols, row_indices=row_indices)
        deltas[col] = full_r2 - reduced_r2 if np.isfinite(reduced_r2) else np.nan
    return full_r2, n_used, deltas


def _save_bootstrap_delta_r2_importance(z, x_cols, output_dir, n_bootstrap=5000, random_seed=20260508):
    raw_feature_names = {
        'Relative_Projection_Residual_z': 'Relative_Projection_Residual',
        'Condition_Number_z': 'Condition_Number',
        'max_abs_loading_z': 'max_abs_loading',
        'pivot_order_z': 'pivot_order',
    }
    feature_labels = {
        'Relative_Projection_Residual': 'RPR',
        'Condition_Number': 'Condition',
        'max_abs_loading': 'Max loading',
        'pivot_order': 'QR rank',
    }

    base_full_r2, _, base_deltas = _dropout_leave_one_variable_out_delta_r2(z, x_cols)

    rng = np.random.default_rng(random_seed)
    sample_rows = []
    attempts = 0
    max_attempts = n_bootstrap * 4
    n = len(z)
    while len(sample_rows) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, n, size=n)
        full_r2, n_used, deltas = _dropout_leave_one_variable_out_delta_r2(
            z,
            x_cols,
            row_indices=indices,
        )
        if not np.isfinite(full_r2):
            continue

        row = {
            'full_r2': full_r2,
            'n_used': n_used,
            'bootstrap': len(sample_rows),
        }
        for col in x_cols:
            raw_name = raw_feature_names[col]
            row[raw_name] = deltas[col]
        sample_rows.append(row)

    samples = pd.DataFrame(sample_rows)
    samples_path = os.path.join(output_dir, 'dropout_bootstrap_delta_r2_samples.csv')
    samples.to_csv(samples_path, index=False)

    summary_rows = []
    for col in x_cols:
        raw_name = raw_feature_names[col]
        values = samples[raw_name].to_numpy(dtype=float) if raw_name in samples else np.array([], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary_rows.append({
            'feature': feature_labels[raw_name],
            'raw_feature': raw_name,
            'base_full_r2': base_full_r2,
            'base_delta_r2': base_deltas[col],
            'bootstrap_median_delta_r2': float(np.nanmedian(values)),
            'ci_2.5': float(np.nanpercentile(values, 2.5)),
            'ci_97.5': float(np.nanpercentile(values, 97.5)),
            'ci_5': float(np.nanpercentile(values, 5)),
            'ci_95': float(np.nanpercentile(values, 95)),
            'prob_delta_r2_gt_0.01': float(np.mean(values > 0.01)),
            'prob_delta_r2_gt_0.05': float(np.mean(values > 0.05)),
            'n_bootstrap': int(values.size),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        'bootstrap_median_delta_r2',
        ascending=False,
    )
    summary_path = os.path.join(output_dir, 'dropout_bootstrap_delta_r2_importance.csv')
    summary.to_csv(summary_path, index=False)
    return samples_path, summary_path


def _save_dropout_model_contribution_figure(
        z,
        full_r2,
        pivot_r2,
        output_dir,
        pivot_rpr_r2=None,
        pivot_rpr_condition_r2=None):
    pivot_color = '#5aa3c7'
    pivot_rpr_color = '#6eb071'
    pivot_rpr_condition_color = '#9b82c0'
    full_color = '#de6b5b'

    fig = plt.figure(figsize=(11.2, 6.7), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.15], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    scatter_specs = [
        ('pred_pivot_z', pivot_color, '^', 'QR-only model'),
        ('pred_pivot_rpr_z', pivot_rpr_color, 's', 'QR + RPR model'),
        ('pred_pivot_rpr_condition_z', pivot_rpr_condition_color, 'D', 'QR + RPR + K model'),
        ('pred_z', full_color, 'o', 'Full model'),
    ]
    present_pred_cols = []
    for pred_col, color, marker, label in scatter_specs:
        if pred_col not in z:
            continue
        present_pred_cols.append(pred_col)
        ax.scatter(
            z[pred_col],
            z['Delta_System_Level_Mean_NSE_z'],
            color=color,
            marker=marker,
            s=50,
            edgecolors='white',
            linewidths=0.55,
            alpha=1.0,
            label=label
        )
    lim_min = min([z[col].min() for col in present_pred_cols] + [z['Delta_System_Level_Mean_NSE_z'].min()])
    lim_max = max([z[col].max() for col in present_pred_cols] + [z['Delta_System_Level_Mean_NSE_z'].max()])
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color='#333333',
        linewidth=1.4,
        linestyle='--',
        dashes=(5, 3),
        label='Perfect line',
        zorder=1
    )
    ax.set_xlabel('Fitted dropout loss (within-r z)', fontsize=12)
    ax.set_ylabel('Actual dropout loss (within-r z)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.18)
    def _adjusted_r2(raw_r2, n_obs, n_predictors):
        if not np.isfinite(raw_r2) or n_obs <= n_predictors + 1:
            return np.nan
        return 1.0 - (1.0 - raw_r2) * (n_obs - 1) / (n_obs - n_predictors - 1)

    n_obs = len(z)
    pivot_adj_r2 = _adjusted_r2(pivot_r2, n_obs, 1)
    pivot_rpr_adj_r2 = _adjusted_r2(pivot_rpr_r2, n_obs, 2) if pivot_rpr_r2 is not None else None
    pivot_rpr_condition_adj_r2 = (
        _adjusted_r2(pivot_rpr_condition_r2, n_obs, 3)
        if pivot_rpr_condition_r2 is not None else None
    )
    full_adj_r2 = _adjusted_r2(full_r2, n_obs, 4)

    r2_rows = [('QR', f'{pivot_adj_r2:.3f}')]
    if pivot_rpr_r2 is not None:
        r2_rows.append(('QR + RPR', f'{pivot_rpr_adj_r2:.3f}'))
    if pivot_rpr_condition_r2 is not None:
        r2_rows.append(('QR + RPR + K', f'{pivot_rpr_condition_adj_r2:.3f}'))
    r2_rows.append(('Full', f'{full_adj_r2:.3f}'))
    header_y = 0.958
    y0 = 0.925
    dy = 0.044
    model_x = 0.02
    r2_x = 0.315
    table_fontsize = 9.0
    ax.text(
        model_x,
        header_y,
        'Model',
        transform=ax.transAxes,
        va='baseline',
        ha='left',
        fontsize=table_fontsize,
        fontweight='bold',
        bbox=dict(facecolor='none', alpha=0.0, edgecolor='none')
    )
    ax.text(
        r2_x,
        header_y,
        'Adjusted R²',
        transform=ax.transAxes,
        va='baseline',
        ha='right',
        fontsize=table_fontsize,
        fontweight='bold',
        bbox=dict(facecolor='none', alpha=0.0, edgecolor='none')
    )
    for idx, (model_label, r2_label) in enumerate(r2_rows):
        y_pos = y0 - idx * dy
        ax.text(
            model_x,
            y_pos,
            model_label,
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=table_fontsize,
            fontweight='normal',
            bbox=dict(facecolor='none', alpha=0.0, edgecolor='none')
        )
        ax.text(
            r2_x,
            y_pos,
            r2_label,
            transform=ax.transAxes,
            va='top',
            ha='right',
            fontsize=table_fontsize,
            fontweight='normal',
            bbox=dict(facecolor='none', alpha=0.0, edgecolor='none')
        )
    ax.text(0.02, 0.025, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        borderaxespad=0.08,
        fontsize=9.5,
        frameon=False
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    ax = fig.add_subplot(gs[0, 1])
    feature_summary = _load_bootstrap_delta_r2_summary(output_dir)
    if feature_summary is None:
        ax.text(0.5, 0.5, 'Bootstrap ΔR² summary\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        y = np.arange(len(feature_summary))
        med = feature_summary['median'].to_numpy(dtype=float)
        lo = feature_summary['ci_5'].to_numpy(dtype=float)
        hi = feature_summary['ci_95'].to_numpy(dtype=float)
        bar_palette_bottom_to_top = _load_dropout_multifactor_palette()['bar_palette_bottom_to_top']
        bar_colors = bar_palette_bottom_to_top[:len(feature_summary)]
        ax.barh(y, med, color=bar_colors, edgecolor='#4a4a4a', linewidth=0.8, alpha=1.0)
        ax.errorbar(
            med,
            y,
            xerr=[np.maximum(med - lo, 0), np.maximum(hi - med, 0)],
            fmt='none',
            ecolor='#222222',
            elinewidth=1.1,
            capsize=3,
            zorder=3
        )
        x_limit = max(0.01, float(np.nanmax(hi)) * 1.25)
        ax.set_xlim(0, x_limit)
        ax.set_yticks(y)
        ax.set_yticklabels(feature_summary['feature'], fontsize=10)
        ax.set_xlabel('Bootstrap leave-one-variable-out $\\Delta R^2$', fontsize=11)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(True, axis='x', alpha=0.18)
        for yi, row in enumerate(feature_summary.itertuples(index=False)):
            ax.text(
                min(row.median + x_limit * 0.025, x_limit * 0.96),
                yi,
                f'{row.relative_contribution * 100:.0f}%',
                va='center',
                ha='left',
                fontsize=9.5
            )
        ax.text(0.99, 0.02, '90% bootstrap interval', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9.5)
    ax.text(0.97, 0.98, 'b', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    fig.tight_layout()
    output_path = os.path.join(output_dir, 'dropout_pivot_vs_full_model_contribution.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _compute_loocv_r2(z, x_cols):
    y = z['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(z))] + [z[col].to_numpy(dtype=float) for col in x_cols])
    pred = np.full(len(z), np.nan, dtype=float)
    for i in range(len(z)):
        mask = np.ones(len(z), dtype=bool)
        mask[i] = False
        coef = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        pred[i] = X[i] @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _compute_dropout_nested_model_r2_summary(z, model_specs):
    rows = []
    n = len(z)
    y = z['Delta_System_Level_Mean_NSE_z'].to_numpy(dtype=float)
    for label, cols in model_specs:
        X = np.column_stack([np.ones(n)] + [z[col].to_numpy(dtype=float) for col in cols])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        raw_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        p = len(cols)
        adj_r2 = 1.0 - (1.0 - raw_r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
        loocv_r2 = _compute_loocv_r2(z, cols)
        rows.append({
            'model': label,
            'n_predictors': p,
            'raw_r2': raw_r2,
            'adjusted_r2': adj_r2,
            'loocv_r2': loocv_r2,
        })
    return pd.DataFrame(rows)


def _save_dropout_nested_model_r2_diagnostics(z, model_specs, output_dir):
    summary = _compute_dropout_nested_model_r2_summary(z, model_specs)

    csv_path = os.path.join(output_dir, 'dropout_nested_model_r2_diagnostics.csv')
    summary.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.3), dpi=300, sharey=True)
    colors = _load_dropout_multifactor_palette()['nested_model_colors']
    x = np.arange(len(summary))
    labels = summary['model'].tolist()
    for ax, metric, ylabel, tag in [
        (axes[0], 'raw_r2', 'Raw $R^2$', 'a'),
        (axes[1], 'adjusted_r2', 'Adjusted $R^2$', 'b'),
        (axes[2], 'loocv_r2', 'Leave-one-out cross-validation $R^2$', 'c'),
    ]:
        values = summary[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=colors, edgecolor='#4a4a4a', linewidth=0.8, alpha=0.88)
        ax.plot(x, values, color='#333333', linewidth=1.0, marker='o', markersize=3.8, zorder=3)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.025, f'{value:.3f}', ha='center', va='bottom', fontsize=8.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=8.8)
        ax.set_ylabel(ylabel, fontsize=10.2)
        ax.set_ylim(0, 0.78)
        ax.grid(True, axis='y', alpha=0.18)
        ax.text(0.02, 0.96, tag, transform=ax.transAxes, ha='left', va='top',
                fontsize=11, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor('#4a4a4a')
    fig.tight_layout(w_pad=1.2)
    output_path = os.path.join(output_dir, 'dropout_nested_model_r2_diagnostics.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path, csv_path


def _save_dropout_model_contribution_with_r2_diagnostics(
        z,
        model_specs,
        output_dir,
        use_backup_palette=False):
    feature_summary = _load_bootstrap_delta_r2_summary(output_dir)
    r2_summary = _compute_dropout_nested_model_r2_summary(z, model_specs)
    palette = _load_dropout_multifactor_palette(use_backup=use_backup_palette)
    colors = palette['model_colors']

    fig = plt.figure(figsize=(12.2, 11.8), dpi=300)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.12], hspace=0.12)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.15, 1.15], wspace=0.22)
    bottom = outer[1].subgridspec(1, 3, wspace=0.22)

    ax = fig.add_subplot(top[0, 0])
    scatter_specs = [
        ('pred_pivot_z', colors['Pivot-only model'], '^', 'QR-only model'),
        ('pred_pivot_rpr_z', colors['Pivot + RPR model'], 's', 'QR + RPR model'),
        ('pred_pivot_rpr_condition_z', colors['Pivot + RPR + K model'], 'D', 'QR + RPR + K model'),
        ('pred_z', colors['Full model'], 'o', 'Full model'),
    ]
    present_pred_cols = []
    for pred_col, color, marker, label in scatter_specs:
        if pred_col not in z:
            continue
        present_pred_cols.append(pred_col)
        ax.scatter(
            z[pred_col],
            z['Delta_System_Level_Mean_NSE_z'],
            color=color,
            marker=marker,
            s=42,
            edgecolors='white',
            linewidths=0.5,
            alpha=1.0,
            label=label
        )
    lim_min = min([z[col].min() for col in present_pred_cols] + [z['Delta_System_Level_Mean_NSE_z'].min()])
    lim_max = max([z[col].max() for col in present_pred_cols] + [z['Delta_System_Level_Mean_NSE_z'].max()])
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color='#333333',
        linewidth=1.25,
        linestyle='--',
        dashes=(5, 3),
        label='Perfect line',
        zorder=1
    )
    ax.set_xlabel('Fitted dropout loss (within-r z)', fontsize=10.8)
    ax.set_ylabel('Actual dropout loss (within-r z)', fontsize=10.8)
    ax.tick_params(axis='both', labelsize=9.2)
    ax.grid(True, alpha=0.18)

    header_y = 0.958
    y0 = 0.925
    dy = 0.044
    model_x = 0.02
    r2_x = 0.25#0.315
    table_fontsize = 8.4
    ax.text(model_x, header_y, 'Model', transform=ax.transAxes, va='baseline',
            ha='left', fontsize=table_fontsize, fontweight='bold')
    ax.text(r2_x, header_y, 'Adjusted R²', transform=ax.transAxes, va='baseline',
            ha='right', fontsize=table_fontsize, fontweight='bold')
    table_model_labels = ['QR', 'QR + RPR', 'QR + RPR + K', 'Full']
    for idx, row in r2_summary.iterrows():
        y_pos = y0 - idx * dy
        ax.text(model_x, y_pos, table_model_labels[idx], transform=ax.transAxes, va='top',
                ha='left', fontsize=table_fontsize)
        ax.text(r2_x, y_pos, f"{row['adjusted_r2']:.3f}", transform=ax.transAxes,
                va='top', ha='right', fontsize=table_fontsize)
    ax.text(0.02, 0.025, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), borderaxespad=0.08,
              fontsize=8.7, frameon=False)

    ax = fig.add_subplot(top[0, 1])
    if feature_summary is None:
        ax.text(0.5, 0.5, 'Bootstrap ΔR² summary\nnot available', ha='center',
                va='center', transform=ax.transAxes, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        y = np.arange(len(feature_summary))
        med = feature_summary['median'].to_numpy(dtype=float)
        lo = feature_summary['ci_5'].to_numpy(dtype=float)
        hi = feature_summary['ci_95'].to_numpy(dtype=float)
        bar_palette_bottom_to_top = palette['bar_palette_bottom_to_top']
        bar_colors = bar_palette_bottom_to_top[:len(feature_summary)]
        ax.barh(y, med, color=bar_colors, edgecolor='#4a4a4a', linewidth=0.75, alpha=1.0)
        ax.errorbar(
            med,
            y,
            xerr=[np.maximum(med - lo, 0), np.maximum(hi - med, 0)],
            fmt='none',
            ecolor='#222222',
            elinewidth=1.0,
            capsize=2.8,
            zorder=3
        )
        x_limit = max(0.01, float(np.nanmax(hi)) * 1.25)
        ax.set_xlim(0, x_limit)
        ax.set_yticks(y)
        ax.set_yticklabels(feature_summary['feature'], fontsize=9.2)
        ax.set_xlabel('Bootstrap leave-one-variable-out $\\Delta R^2$', fontsize=9.5)
        ax.tick_params(axis='x', labelsize=9.0)
        ax.grid(True, axis='x', alpha=0.18)
        for yi, row in enumerate(feature_summary.itertuples(index=False)):
            ax.text(min(row.median + x_limit * 0.025, x_limit * 0.96), yi,
                    f'{row.relative_contribution * 100:.0f}%', va='center',
                    ha='left', fontsize=8.8)
        ax.text(0.98, 0.01, '90% bootstrap interval', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.8)
    ax.text(0.97, 0.98, 'b', transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold')

    diagnostic_specs = [
        ('raw_r2', 'Raw $R^2$', 'c'),
        ('adjusted_r2', 'Adjusted $R^2$', 'd'),
        ('loocv_r2', 'Leave-one-out cross-validation $R^2$', 'e'),
    ]
    x = np.arange(len(r2_summary))
    model_labels = ['QR', '+RPR', '+K', '+L\n(Full)']
    model_colors = palette['nested_model_colors']
    for idx, (metric, ylabel, tag) in enumerate(diagnostic_specs):
        ax = fig.add_subplot(bottom[0, idx])
        values = r2_summary[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=model_colors, edgecolor='#4a4a4a', linewidth=0.75, alpha=1.0)
        ax.plot(x, values, color='#333333', linewidth=0.95, marker='o', markersize=3.4, zorder=3)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.025, f'{value:.3f}', ha='center', va='bottom', fontsize=7.8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=0, ha='center', fontsize=9.0)
        ax.set_ylabel(ylabel, fontsize=10.6)
        ax.tick_params(axis='y', labelsize=9.5)
        ax.set_ylim(0, 0.82)
        ax.grid(True, axis='y', alpha=0.18)
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, ha='left', va='top',
                fontsize=10, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
            spine.set_edgecolor('#4a4a4a')

    fig.text(0.5, 0.066, 'Nested explanatory model', ha='center', va='center',
             fontsize=10.8)

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
            spine.set_edgecolor('#4a4a4a')

    output_name = (
        'dropout_model_contribution_with_r2_diagnostics_backup_palette.png'
        if use_backup_palette
        else 'dropout_model_contribution_with_r2_diagnostics.png'
    )
    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _save_dropout_multifactor_summary(z, x_cols, coef, r2, reduced_r2, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), dpi=300)

    ax = axes[0, 0]
    scatter = ax.scatter(
        z['Relative_Projection_Residual_z'],
        z['Delta_System_Level_Mean_NSE_z'],
        c=z['Condition_Number_z'],
        cmap='viridis',
        s=58,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.95
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Condition number (within-r z)')
    ax.set_xlabel('Relative projection residual (within-r z)')
    ax.set_ylabel('Delta System-level mean NSE (within-r z)')
    ax.grid(True, alpha=0.18)
    corr = z['Relative_Projection_Residual_z'].corr(z['Delta_System_Level_Mean_NSE_z'])
    ax.text(0.02, 0.97, f'corr = {corr:.3f}', transform=ax.transAxes, va='top',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='#999999'))
    ax.text(0.02, 0.02, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    ax = axes[0, 1]
    scatter = ax.scatter(
        z['Condition_Number_z'],
        z['Delta_System_Level_Mean_NSE_z'],
        c=z['max_abs_loading_z'],
        cmap='coolwarm',
        s=58,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.95
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Max modal loading (within-r z)')
    ax.set_xlabel('Condition number (within-r z)')
    ax.set_ylabel('Delta System-level mean NSE (within-r z)')
    ax.grid(True, alpha=0.18)
    corr = z['Condition_Number_z'].corr(z['Delta_System_Level_Mean_NSE_z'])
    ax.text(0.02, 0.97, f'corr = {corr:.3f}', transform=ax.transAxes, va='top',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='#999999'))
    ax.text(0.02, 0.02, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    ax = axes[1, 0]
    full_color = '#d65244'
    reduced_color = '#4a98c5'
    ax.scatter(
        z['pred_reduced_z'],
        z['Delta_System_Level_Mean_NSE_z'],
        color=reduced_color,
        marker='^',
        s=54,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.88,
        label='RPR + condition'
    )
    ax.scatter(
        z['pred_z'],
        z['Delta_System_Level_Mean_NSE_z'],
        color=full_color,
        marker='o',
        s=50,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.88,
        label='Full model'
    )
    lim_min = min(
        z['pred_reduced_z'].min(),
        z['pred_z'].min(),
        z['Delta_System_Level_Mean_NSE_z'].min()
    )
    lim_max = max(
        z['pred_reduced_z'].max(),
        z['pred_z'].max(),
        z['Delta_System_Level_Mean_NSE_z'].max()
    )
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color='#333333',
        linewidth=1.5,
        linestyle='--',
        dashes=(5, 3),
        label='Perfect line',
        zorder=1
    )
    ax.set_xlabel('Fitted dropout loss (within-r z)')
    ax.set_ylabel('Actual dropout loss (within-r z)')
    ax.grid(True, alpha=0.18)
    ax.text(0.02, 0.97, f'2-var $R^2$ = {reduced_r2:.3f}\nFull $R^2$ = {r2:.3f}', transform=ax.transAxes, va='top',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='#999999'))
    _add_dropout_residual_inset(ax, z, reduced_color=reduced_color, full_color=full_color)
    ax.legend(loc='lower right', fontsize=9.0, frameon=True, ncol=1)
    ax.text(0.02, 0.02, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    ax = axes[1, 1]
    coef_names = ['RPR', 'Condition', 'Max loading', 'QR rank']
    coef_values = coef[1:]
    colors = ['#4a98c5', '#58a55c', '#d65244', '#8b6bb3']
    ax.bar(coef_names, coef_values, color=colors, edgecolor='#4a4a4a', linewidth=0.8)
    ax.axhline(0.0, color='#333333', linewidth=1.0)
    ax.set_ylabel('Standardized coefficient')
    ax.grid(True, axis='y', alpha=0.18)
    ax.text(0.02, 0.02, '(d)', transform=ax.transAxes, fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#4a4a4a')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dropout_multifactor_summary.png'), dpi=300)
    plt.close()

    coef_df = pd.DataFrame({
        'feature': ['intercept'] + x_cols,
        'coefficient': coef
    })
    coef_df.to_csv(os.path.join(output_dir, 'dropout_multifactor_standardized_coefficients.csv'), index=False)


def generate_dropout_multifactor_analysis(psi, current_dir=None, normalization_mode='global_minmax'):
    _, output_dir, _ = _dropout_base_paths(current_dir=current_dir, normalization_mode=normalization_mode)
    os.makedirs(output_dir, exist_ok=True)
    df = _load_dropout_multifactor_augmented_data(psi, current_dir=current_dir, normalization_mode=normalization_mode)
    z, x_cols, coef, r2 = _fit_dropout_multifactor_standardized_model(df)
    z, x_cols_pivot, coef_pivot, pivot_r2 = _fit_dropout_pivot_only_standardized_model(z)
    z, _, _, pivot_rpr_r2 = _fit_dropout_named_standardized_model(
        z,
        ['pivot_order_z', 'Relative_Projection_Residual_z'],
        'pred_pivot_rpr_z'
    )
    z, _, _, pivot_rpr_condition_r2 = _fit_dropout_named_standardized_model(
        z,
        ['pivot_order_z', 'Relative_Projection_Residual_z', 'Condition_Number_z'],
        'pred_pivot_rpr_condition_z'
    )
    z, x_cols_reduced, coef_reduced, reduced_r2 = _fit_dropout_reduced_standardized_model(z)
    _save_bootstrap_delta_r2_importance(z, x_cols, output_dir)
    _save_dropout_multifactor_summary(z, x_cols, coef, r2, reduced_r2, output_dir)
    contribution_path = _save_dropout_model_contribution_figure(
        z,
        r2,
        pivot_r2,
        output_dir,
        pivot_rpr_r2=pivot_rpr_r2,
        pivot_rpr_condition_r2=pivot_rpr_condition_r2
    )
    model_specs = [
        ('QR', ['pivot_order_z']),
        ('QR + RPR', ['pivot_order_z', 'Relative_Projection_Residual_z']),
        ('QR + RPR + K', ['pivot_order_z', 'Relative_Projection_Residual_z', 'Condition_Number_z']),
        ('Full', ['pivot_order_z', 'Relative_Projection_Residual_z', 'Condition_Number_z', 'max_abs_loading_z']),
    ]
    r2_diagnostics_path, r2_diagnostics_csv = _save_dropout_nested_model_r2_diagnostics(
        z,
        model_specs,
        output_dir
    )
    contribution_r2_path = _save_dropout_model_contribution_with_r2_diagnostics(
        z,
        model_specs,
        output_dir
    )
    return {
        'summary': os.path.join(output_dir, 'dropout_multifactor_summary.png'),
        'pivot_vs_full_contribution': contribution_path,
        'pivot_vs_full_contribution_r2': contribution_r2_path,
        'r2_diagnostics': r2_diagnostics_path,
        'r2_diagnostics_csv': r2_diagnostics_csv,
        'coefficients': os.path.join(output_dir, 'dropout_multifactor_standardized_coefficients.csv'),
    }


def _load_benchmark_cache(normalization_mode='global_minmax', model_dir=None):
    base_dir = model_dir or os.path.join(os.path.dirname(__file__), 'saved_models')
    path = os.path.join(base_dir, f'random_benchmark_eventwise_{normalization_mode}.npz')
    data = np.load(path, allow_pickle=True)
    return {
        'dss_eventwise': data['dss_eventwise'].item(),
        'exhaustive_distribution': data['exhaustive_distribution'].item(),
        'exhaustive_optimum': data['exhaustive_optimum'].item(),
        'monte_carlo_distribution': data['monte_carlo_distribution'].item(),
    }


def _stitch_benchmark_side_by_side(left_path, right_path, out_path, gap_px=30, margin_px=22):
    left = Image.open(left_path).convert('RGB')
    right = Image.open(right_path).convert('RGB')

    def _autocrop(img, threshold=250):
        arr = np.array(img.convert('L'))
        mask = arr < threshold
        if not mask.any():
            return img
        ys, xs = np.where(mask)
        return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    left = _autocrop(left)
    right = _autocrop(right)
    left = ImageOps.expand(left, border=(12, 12, 12, 12), fill='white')
    right = ImageOps.expand(right, border=(12, 12, 12, 12), fill='white')

    target_height = max(left.height, right.height)

    def _resize_keep_height(img, height):
        if img.height == height:
            return img
        width = int(round(img.width * height / img.height))
        return img.resize((width, height), Image.LANCZOS)

    left = _resize_keep_height(left, target_height)
    right = _resize_keep_height(right, target_height)

    canvas = Image.new(
        'RGB',
        (left.width + right.width + gap_px + 2 * margin_px, target_height + 2 * margin_px),
        'white'
    )
    canvas.paste(left, (margin_px, margin_px))
    canvas.paste(right, (margin_px + left.width + gap_px, margin_px))
    canvas.save(out_path, quality=95)


def create_benchmark_paper_composite(normalization_mode='global_minmax', current_dir=None, model_dir=None):
    base_dir = current_dir or os.path.dirname(__file__)
    model_dir = model_dir or os.path.join(base_dir, 'saved_models')

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'

    cache = _load_benchmark_cache(normalization_mode, model_dir)

    exhaustive_distribution = {r: cache['exhaustive_distribution'][r] for r in sorted(cache['exhaustive_distribution']) if r <= 4}
    dss_exhaustive = {r: cache['dss_eventwise'][r] for r in sorted(cache['dss_eventwise']) if r <= 4}
    optimum = {r: cache['exhaustive_optimum'][r] for r in sorted(cache['exhaustive_optimum']) if r <= 4}

    random_distribution = {
        r: [float(v) for sample in cache['monte_carlo_distribution'][r] for v in sample]
        for r in sorted(cache['monte_carlo_distribution']) if r >= 5
    }
    dss_random = {r: cache['dss_eventwise'][r] for r in sorted(cache['dss_eventwise']) if r >= 5}

    left_path = os.path.join(base_dir, f'Benchmark_Exhaustive_vs_DSS_{normalization_mode}.png')
    right_path = os.path.join(base_dir, f'Benchmark_Random_MC_vs_DSS_{normalization_mode}.png')
    combo_path = os.path.join(base_dir, f'Benchmark_paper_composite_{normalization_mode}.png')

    plot_exhaustive_eventwise_benchmark(
        exhaustive_distribution,
        dss_exhaustive,
        optimum,
        base_dir,
        output_filename=os.path.basename(left_path),
        axis_segments=[(0.80, 1.00), (-1.2, 0.80), (-5.0, -1.2)],
        show_panel_labels=True,
    )
    plot_random_monte_carlo_eventwise_benchmark(
        random_distribution,
        dss_random,
        base_dir,
        output_filename=os.path.basename(right_path),
        title_suffix='',
        show_panel_label=True,
    )

    axis_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 11
    panel_tag_fontsize = 16
    note_fontsize = 16
    fig = plt.figure(figsize=(14.2, 7.6), dpi=300)
    outer = GridSpec(1, 2, width_ratios=[1.02, 1.0], wspace=0.14)

    # Left: exhaustive benchmark, 3 broken-axis panels
    left_grid = outer[0, 0].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)
    left_axes = [fig.add_subplot(left_grid[idx, 0]) for idx in range(3)]
    left_r = sorted(exhaustive_distribution.keys())
    left_positions = np.arange(1, len(left_r) + 1, dtype=float)
    dss_positions = left_positions - 0.28
    optimum_positions = left_positions + 0.28
    left_distributions = [exhaustive_distribution[r] for r in left_r]
    left_segments = [(0.80, 1.01), (-1.0, 0.79), (-5.0, -1.2)]
    box_handle = None
    for idx, ax in enumerate(left_axes):
        box = _draw_distribution_boxes(ax, left_positions, left_distributions, '#3B5DA3', label='All combinations', widths=0.28)
        if box_handle is None:
            box_handle = box['boxes'][0]
        _draw_event_points(ax, dss_positions, dss_exhaustive, '#f8dfa6', 'o', label='DSS')
        _draw_event_points(ax, optimum_positions, optimum, '#6CB3DA', 's', label='Exhaustive optimum')
        ax.set_ylim(*left_segments[idx])
        if idx == 0:
            ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    _style_broken_axis_stack(left_axes, show_xlabel=True, label_size=axis_fontsize, tick_size=tick_fontsize)
    left_axes[-1].set_xticks(left_positions)
    left_axes[-1].set_xticklabels([str(r) for r in left_r], fontsize=tick_fontsize)
    left_axes[0].text(0.010, 0.96, 'a', transform=left_axes[0].transAxes, fontsize=panel_tag_fontsize, fontweight='bold', va='top')
    left_axes[1].set_ylabel('System-level NSE', fontsize=axis_fontsize)
    left_axes[1].yaxis.set_label_coords(-0.09, 0.5)

    # Right: random benchmark, 3 broken-axis panels
    right_grid = outer[0, 1].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)
    right_axes = [fig.add_subplot(right_grid[idx, 0]) for idx in range(3)]
    right_r = sorted(random_distribution.keys())
    right_positions = np.arange(1, len(right_r) + 1, dtype=float)
    right_distributions = [random_distribution[r] for r in right_r]
    right_segments = [(0.75, 1.01), (-4.0, 0.74), (-15.0, -5.0)]
    right_box_handle = None
    for idx, ax in enumerate(right_axes):
        box = _draw_distribution_boxes(ax, right_positions, right_distributions, '#aa2b46', label='Random', widths=0.44)
        if right_box_handle is None:
            right_box_handle = box['boxes'][0]
        _draw_event_points(ax, right_positions, dss_random, '#f8dfa6', 'o', label='DSS', offset_width=0.30)
        ax.set_ylim(*right_segments[idx])
        if idx == 0:
            ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        elif idx == 1:
            ax.set_yticks([-4, -3, -2, -1, 0])
        else:
            ax.set_yticks([-14, -12, -10, -8, -6])
        if idx == 0:
            ax.text(0.010, 0.96, 'b', transform=ax.transAxes, fontsize=panel_tag_fontsize, fontweight='bold', va='top')
    _style_broken_axis_stack(right_axes, show_xlabel=True, label_size=axis_fontsize, tick_size=tick_fontsize)
    right_axes[-1].set_xticks(right_positions)
    right_axes[-1].set_xticklabels([str(r) for r in right_r], fontsize=tick_fontsize)
    right_axes[-1].tick_params(axis='y', labelsize=tick_fontsize)
    right_axes[-1].set_xlabel('Number of Sensors (r)', fontsize=axis_fontsize)
    left_axes[-1].legend(
        [
            box_handle,
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f8dfa6', markeredgecolor='#f8dfa6', markersize=7),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#6CB3DA', markeredgecolor='#6CB3DA', markersize=7),
            right_box_handle,
        ],
        ['All combinations', 'DSS', 'Exhaustive optimum', 'Random'],
        fontsize=legend_fontsize,
        loc='lower left',
        bbox_to_anchor=(0.005, 0.015),
        frameon=False,
        ncol=1,
        handletextpad=0.55,
        borderpad=0.35,
        labelspacing=0.35
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.975, bottom=0.10)
    plt.savefig(combo_path, bbox_inches='tight')
    plt.close(fig)
    return left_path, right_path, combo_path


def plot_normalization_performance_comparison(summary_df, current_dir, output_filename='Normalization_Comparison_Performance.png'):
    blue = '#3C78A8'
    red = '#D65244'
    dark = '#555555'

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.4), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.16)

    ax = axes[0]
    ax.plot(summary_df['r'], summary_df['gm_obs_mean'], color=blue, marker='o', markersize=4.5, linewidth=1.9, label='Global-MinMax, observed')
    ax.plot(summary_df['r'], summary_df['zs_obs_mean'], color=red, marker='o', markersize=4.5, linewidth=1.9, label='Node-level z-score, observed')
    ax.plot(summary_df['r'], summary_df['gm_200year'], color=blue, linestyle='--', linewidth=1.6, label='Global-MinMax, 200-year')
    ax.plot(summary_df['r'], summary_df['zs_200year'], color=red, linestyle='--', linewidth=1.6, label='Node-level z-score, 200-year')
    ax.set_ylabel('System-level NSE', fontsize=12)
    ax.set_ylim(0.65, 1.01)
    ax.legend(fontsize=8.5, loc='lower right', frameon=False, facecolor='none', borderpad=0.35, handlelength=2.2)
    ax.text(0.015, 0.96, 'a', transform=ax.transAxes, fontsize=11.5, fontweight='bold', va='top')

    ax = axes[1]
    gm_err = np.vstack([
        summary_df['gm_node_obs_median'].to_numpy() - summary_df['gm_node_obs_q1'].to_numpy(),
        summary_df['gm_node_obs_q3'].to_numpy() - summary_df['gm_node_obs_median'].to_numpy()
    ])
    zs_err = np.vstack([
        summary_df['zs_node_obs_median'].to_numpy() - summary_df['zs_node_obs_q1'].to_numpy(),
        summary_df['zs_node_obs_q3'].to_numpy() - summary_df['zs_node_obs_median'].to_numpy()
    ])
    ax.errorbar(summary_df['r'].to_numpy() - 0.06, summary_df['gm_node_obs_median'].to_numpy(), yerr=gm_err, fmt='o-', color=blue, linewidth=1.8, elinewidth=1.0, capsize=3, markersize=4.0, label='Global-MinMax, observed median')
    ax.errorbar(summary_df['r'].to_numpy() + 0.06, summary_df['zs_node_obs_median'].to_numpy(), yerr=zs_err, fmt='o-', color=red, linewidth=1.8, elinewidth=1.0, capsize=3, markersize=4.0, label='Node-level z-score, observed median')
    ax.plot(summary_df['r'], summary_df['gm_node_200year_median'], color=blue, linestyle='--', linewidth=1.5, label='Global-MinMax, 200-year')
    ax.plot(summary_df['r'], summary_df['zs_node_200year_median'], color=red, linestyle='--', linewidth=1.5, label='Node-level z-score, 200-year')
    ax.set_ylabel('Node-level NSE', fontsize=12)
    ax.set_ylim(0.0, 1.01)
    ax.legend(fontsize=8.3, loc='lower right', frameon=False, facecolor='none', borderpad=0.35, handlelength=2.2)
    ax.text(0.015, 0.96, 'b', transform=ax.transAxes, fontsize=11.5, fontweight='bold', va='top')

    ax = axes[2]
    ax.plot(summary_df['r'], summary_df['overlap_fraction'], color=dark, marker='s', markersize=4.0, linewidth=1.8)
    for _, row in summary_df.iterrows():
        ax.text(row['r'], row['overlap_fraction'] + 0.04, f"{int(row['overlap_count'])}/{int(row['r'])}", ha='center', va='bottom', fontsize=7.5)
    ax.set_ylabel('Selection overlap fraction', fontsize=12)
    ax.set_xlabel('r (Number of sensors used)', fontsize=12)
    ax.set_ylim(-0.02, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.text(0.015, 0.96, 'c', transform=ax.transAxes, fontsize=11.5, fontweight='bold', va='top')

    for ax in axes:
        ax.set_xlim(0.85, 10.15)
        ax.set_xticks(np.arange(1, 11))
        ax.tick_params(axis='both', labelsize=9.5)

    output_path = os.path.join(current_dir, output_filename)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_random_sampling_convergence_summary(summary_df, current_dir, output_filename='Random_Sampling_Convergence.png'):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.4), dpi=300, sharex=True)
    cmap = plt.get_cmap('tab10')
    r_values = sorted(summary_df['r'].unique())

    for idx, r in enumerate(r_values):
        subset = summary_df[summary_df['r'] == r].sort_values('sample_size')
        color = cmap(idx % 10)
        axes[0].plot(subset['sample_size'], subset['max_abs_delta_to_reference'], marker='o', linewidth=1.7, markersize=4.0, color=color, label=f'r={r}')
        axes[1].plot(subset['sample_size'], subset['main_range_across_seeds'], marker='s', linewidth=1.5, markersize=3.8, color=color)
        axes[1].plot(subset['sample_size'], subset['tail_range_across_seeds'], marker='^', linewidth=1.2, markersize=3.8, color=color, linestyle='--')

    axes[0].set_ylabel('Max |delta| to reference', fontsize=12)
    axes[1].set_ylabel('Across-seed range', fontsize=12)
    axes[1].set_xlabel('Monte Carlo sample size', fontsize=12)
    axes[0].legend(fontsize=8.5, ncol=2, frameon=True, loc='upper right')
    axes[0].text(0.015, 0.08, '(a)', transform=axes[0].transAxes, fontsize=11.5, fontweight='bold')
    axes[1].text(0.015, 0.08, '(b)', transform=axes[1].transAxes, fontsize=11.5, fontweight='bold')

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.tick_params(axis='both', labelsize=9.5)

    output_path = os.path.join(current_dir, output_filename)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path


def run_sensor_stability_analysis(Psi, S, current_dir, max_r=10, mode_label='Raw SVD'):
    r_values = range(1, max_r + 1)
    condition_numbers = []
    cumulative_energies = []
    total_energy = np.sum(S ** 2)

    print(f"Sensor stability analysis mode: {mode_label}")
    print(f"{'r (Sensors)':<15} | {'Condition Number (kappa)':<25} | {'Cumulative Energy (%)':<25}")
    print("-" * 70)

    for r in r_values:
        ind_Psi_r = np.arange(r)
        _, _, pivoting = qr(Psi[:, ind_Psi_r].T, pivoting=True)
        Pivot = pivoting[:r]
        Psi_pivot = Psi[Pivot, :][:, ind_Psi_r]

        cond_num = np.linalg.cond(Psi_pivot)
        condition_numbers.append(cond_num)

        cum_energy = np.sum(S[:r] ** 2) / total_energy * 100
        cumulative_energies.append(cum_energy)

        print(f"{r:<15} | {cond_num:<25.2f} | {cum_energy:<24.4f}%")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Number of Sensors (r)', fontsize=14)
    ax1.set_ylabel(r'Condition Number of $\Psi_{pivot}$ (Log Scale)', color=color, fontsize=14)
    ax1.plot(r_values, condition_numbers, marker='s', color=color, linewidth=2, markersize=8, label='Condition Number')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.set_xticks(list(r_values))
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative Energy of POD Modes (%)', color=color, fontsize=14)
    ax2.plot(r_values, cumulative_energies, marker='o', linestyle='--', color=color, linewidth=2, markersize=8,
             label='Cumulative Energy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')

    fig.tight_layout()
    ax1.grid(True, linestyle='--', alpha=0.5)

    output_path = os.path.join(current_dir, 'Sensor_Stability_Analysis.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Stability analysis plot saved to: {output_path}")

    return {
        'r_values': list(r_values),
        'condition_numbers': condition_numbers,
        'cumulative_energies': cumulative_energies,
        'output_path': output_path,
        'mode_label': mode_label,
    }
