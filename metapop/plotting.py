# Visualize incidence and cumulative incidence time curves
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

__all__ = [
    'plot_daily_infectious',
    'plot_interval_incidence',
    'plot_interval_cumulative_incidence',
]

def extract_plot_params(plkwargs):
    return {
        'R0': plkwargs.get('R0', 12),
        'alpha': plkwargs.get('alpha', 1),
        'transparent': plkwargs.get('transparent', False),
        'show_grid': plkwargs.get('show_grid', True),
        'width': plkwargs.get('width', 3.5),
        'height': plkwargs.get('height', 3.3),
        'wspace': plkwargs.get('wspace', 0.1),
        'hspace': plkwargs.get('hspace', 0.1),
        'margins': plkwargs.get('margins', dict(left=0.08, right=0.97, top=0.94, bottom=0.08)),
        'replicates': plkwargs.get('replicates', 20),
        'groups': plkwargs.get('groups', []),
        'colors': plkwargs.get('colors', []),
        'vax_levs': plkwargs.get('vax_levs', []),
        'k_21_vals': plkwargs.get('k_21_vals', []),
        'seed': plkwargs.get('seed', None),
        'figname': plkwargs.get('figname', "figure.png"),
        'dpi': plkwargs.get('dpi', 300),
    }

def plot_daily_infectious(results, **plkwargs):

    params = extract_plot_params(plkwargs)
    alpha = params['alpha']
    transparent = params['transparent']
    show_grid = params['show_grid']
    width = params['width']
    height = params['height']
    wspace = params['wspace']
    hspace = params['hspace']
    margins = params['margins']
    replicates = params['replicates']
    groups = params['groups']
    colors = params['colors']
    vax_levs = params['vax_levs']
    k_21_vals = params['k_21_vals']
    seed = params['seed']
    figname = params['figname'] if params['figname'] != "figure.png" else f"daily_infectious_curves_{params['R0']}.png"
    dpi = params['dpi']

    if seed is not None:
        np.random.seed(seed)

    # define scenarios to plot as subplots
    # define vaccine coverage scenarios if not provided
    if not vax_levs:
        vax_levs = sorted(results['initial_coverage_scenario'].unique().to_list())
    # define connectivity between smaller populations if not provided
    if not k_21_vals:
        k_21_vals = results['k_21'].unique().to_numpy()

    # how many panels needed?
    nrows = len(k_21_vals)
    ncols = len(vax_levs)

    # choose replicates
    replicate_indices = np.random.choice(
        results["replicate"].unique().to_numpy(),
        size=replicates,
        replace=False
    )

    # get some replicates to plot
    filtered_results = results.filter(
        (pl.col("replicate").is_in(replicate_indices))
        & (pl.col("initial_coverage_scenario").is_in(vax_levs))
        & (pl.col("k_21").is_in(k_21_vals))
    )
    filtered_results = filtered_results.with_columns(
        (pl.col('I1') + pl.col('I2')).alias('infectious')
    )

    # plot infectious curves
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(width * ncols, height * nrows),
        sharex=True, sharey=True,
        gridspec_kw=dict(wspace=wspace, hspace=hspace),
    )
    fig.subplots_adjust(**margins)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.expand_dims(axes, axis=0 if nrows == 1 else 1)

    for i, k_21 in enumerate(k_21_vals):
        for j, vax_lev in enumerate(vax_levs):
            ax = axes[i, j]
            for group, color in zip(groups, colors):
                group_data = filtered_results.filter(
                    (pl.col("group") == group)
                    & (pl.col("initial_coverage_scenario") == vax_lev)
                    & (pl.col("k_21") == k_21)
                )
                for replicate in replicate_indices:
                    replicate_data = group_data.filter(pl.col("replicate") == replicate)
                    infectious = replicate_data["infectious"]

                    ax.plot(replicate_data["t"],
                            infectious,
                            label=f'Group {group}, Replicate {replicate}',
                            color=color,
                            alpha=alpha,
                            lw=0.5,
                            zorder=-group
                            )
            if show_grid:
                ax.grid(True)

    # add labels for columns - vaccine coverage
    for j, vax_lev in enumerate(vax_levs):
        axes[0, j].set_title(f"Vaccine coverage: {vax_lev}")
    # add labels for rows - connectivity
    for i, k_21 in enumerate(k_21_vals):
        axes[i, -1].yaxis.set_label_position("right")
        axes[i, -1].set_ylabel(f"Connectivity: {k_21}", rotation=270, labelpad=15)

    fig.supxlabel('Time (Days)')
    fig.supylabel('Infectious')

    figpath = os.path.join("output", figname)
    fig.savefig(figpath, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return fig, axes

def plot_interval_incidence(results, interval=1, **plkwargs):

    params = extract_plot_params(plkwargs)
    alpha = params['alpha']
    transparent = params['transparent']
    show_grid = params['show_grid']
    width = params['width']
    height = params['height']
    wspace = params['wspace']
    hspace = params['hspace']
    margins = params['margins']
    replicates = params['replicates']
    groups = params['groups']
    colors = params['colors']
    vax_levs = params['vax_levs']
    k_21_vals = params['k_21_vals']
    seed = params['seed']
    figname = params['figname'] if params['figname'] != "figure.png" else f"interval_{interval}_incidence_curves_{params['R0']}.png"

    dpi = params['dpi']

    if seed is not None:
        np.random.seed(seed)

    # define scenarios to plot as subplots
    # define vaccine coverage scenarios if not provided
    if not vax_levs:
        vax_levs = sorted(results['initial_coverage_scenario'].unique().to_list())
    # define connectivity between smaller populations if not provided
    if not k_21_vals:
        k_21_vals = results['k_21'].unique().to_numpy()

    # how many panels needed?
    nrows = len(k_21_vals)
    ncols = len(vax_levs)

    # choose replicates
    replicate_indices = np.random.choice(
        results["replicate"].unique().to_numpy(),
        size=replicates,
        replace=False
    )

    # get some replicates to plot
    filtered_results = results.filter(
        (pl.col("replicate").is_in(replicate_indices))
        & (pl.col("initial_coverage_scenario").is_in(vax_levs))
        & (pl.col("k_21").is_in(k_21_vals))
    )

    # plot infectious curves
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(width * ncols, height * nrows),
        sharex=True, sharey=True,
        gridspec_kw=dict(wspace=wspace, hspace=hspace),
    )
    fig.subplots_adjust(**margins)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.expand_dims(axes, axis=0 if nrows == 1 else 1)

    # get time points to plot once
    t = filtered_results["t"].unique().sort().gather_every(interval)
    t = np.arange(len(t))

    for i, k_21 in enumerate(k_21_vals):
        for j, vax_lev in enumerate(vax_levs):
            ax = axes[i, j]
            for group, color in zip(groups, colors):
                group_data = filtered_results.filter(
                    (pl.col("group") == group)
                    & (pl.col("initial_coverage_scenario") == vax_lev)
                    & (pl.col("k_21") == k_21)
                )
                for replicate in replicate_indices:
                    replicate_data = group_data.filter(pl.col("replicate") == replicate)
                    cumulative_incidence = replicate_data["Y"]
                    cumulative_incidence = cumulative_incidence.gather_every(interval)

                    incidence = cumulative_incidence.diff(n = 1).fill_nan(0)
                    ax.plot(t,
                            incidence,
                            label=f'Group {group}, Replicate {replicate}',
                            color=color,
                            alpha=alpha,
                            lw=0.5,
                            zorder=-group
                            )
            if show_grid:
                ax.grid(True)

    # add labels for columns - vaccine coverage
    for j, vax_lev in enumerate(vax_levs):
        axes[0, j].set_title(f"Vaccine coverage: {vax_lev}")
    # add labels for rows - connectivity
    for i, k_21 in enumerate(k_21_vals):
        axes[i, -1].yaxis.set_label_position("right")
        axes[i, -1].set_ylabel(f"Connectivity: {k_21}", rotation=270, labelpad=15)

    fig.supxlabel(f'Time ( {interval} Days)')
    if interval == 1:
        fig.supxlabel('Time (Days)')
    elif interval == 7:
        fig.supxlabel('Time (Weeks)')
    fig.supylabel('Incidence')

    figpath = os.path.join("output", figname)
    fig.savefig(figpath, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return fig, axes

def plot_interval_cumulative_incidence(results, **plkwargs):

    params = extract_plot_params(plkwargs)
    alpha = params['alpha']
    transparent = params['transparent']
    show_grid = params['show_grid']
    width = params['width']
    height = params['height']
    wspace = params['wspace']
    hspace = params['hspace']
    margins = params['margins']
    replicates = params['replicates']
    groups = params['groups']
    colors = params['colors']
    vax_levs = params['vax_levs']
    k_21_vals = params['k_21_vals']
    seed = params['seed']
    figname = params['figname'] if params['figname'] != "figure.png" else f"daily_cumulative_incidence_curves_{params['R0']}.png"
    dpi = params['dpi']

    if seed is not None:
        np.random.seed(seed)

    # define scenarios to plot as subplots
    # define vaccine coverage scenarios if not provided
    if not vax_levs:
        vax_levs = sorted(results['initial_coverage_scenario'].unique().to_list())
    # define connectivity between smaller populations if not provided
    if not k_21_vals:
        k_21_vals = results['k_21'].unique().to_numpy()

    # how many panels needed?
    nrows = len(k_21_vals)
    ncols = len(vax_levs)

    # choose replicates
    replicate_indices = np.random.choice(
        results["replicate"].unique().to_numpy(),
        size=replicates,
        replace=False
    )

    # get some replicates to plot
    filtered_results = results.filter(
        (pl.col("replicate").is_in(replicate_indices))
        & (pl.col("initial_coverage_scenario").is_in(vax_levs))
        & (pl.col("k_21").is_in(k_21_vals))
    )

    # plot incidence curves
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(width * ncols, height * nrows),
        sharex=True, sharey=True,
        gridspec_kw=dict(wspace=wspace, hspace=hspace),
    )
    fig.subplots_adjust(**margins)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.expand_dims(axes, axis=0 if nrows == 1 else 1)

    for i, k_21 in enumerate(k_21_vals):
        for j, vax_lev in enumerate(vax_levs):
            ax = axes[i, j]
            for group, color in zip(groups, colors):
                group_data = filtered_results.filter(
                    (pl.col("group") == group)
                    & (pl.col("initial_coverage_scenario") == vax_lev)
                    & (pl.col("k_21") == k_21)
                )
                for replicate in replicate_indices:
                    replicate_data = group_data.filter(pl.col("replicate") == replicate)

                    cumulative_incidence = replicate_data["Y"]

                    ax.plot(replicate_data["t"],
                            cumulative_incidence,
                            label=f'Group {group}, Replicate {replicate}',
                            color=color,
                            alpha=alpha,
                            lw=0.5,
                            zorder=-group
                            )
            if show_grid:
                ax.grid(True)

    # add labels for columns - vaccine coverage
    for j, vax_lev in enumerate(vax_levs):
        axes[0, j].set_title(f"Vaccine coverage: {vax_lev}")
    # add labels for rows - connectivity
    for i, k_21 in enumerate(k_21_vals):
        axes[i, -1].yaxis.set_label_position("right")
        axes[i, -1].set_ylabel(f"Connectivity: {k_21}", rotation=270, labelpad=15)

    fig.supxlabel('Time (Days)')
    fig.supylabel('Cumulative Incidence')

    figpath = os.path.join("output", figname)
    fig.savefig(figpath, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return fig, axes
