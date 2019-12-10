import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

from runs import eil, RunPesticideExcptEggs, RunPesticideGeneral, RunBiocontrolPupa, DRunBiocontrolPupa, \
    time_range, DRunBiocontrolLarva, RunBiocontrolLarva, ORunBiocontrolPupa3, ORunBiocontrolLarvas3, BaseRun, \
    NoPupalParasto150, NoLarvalParasto150

"""
This is the code used to run all analyses and generate all figures present in the article.
"""

rep_axes = (-1, -2, -3)
ax_time = 1
ax_multi = 0
dir_figs = 'figs'


def slice_time(r, sl, multi=True):
    ax_t = ax_time if multi else ax_time - 1
    return get_x(r, sl, multi=multi), r[[slice(None)] * ax_t + [sl]]


def get_x(r, sl, multi=True):
    ax_t = ax_time if multi else ax_time - 1
    return np.arange(r.shape[ax_t])[sl]


def get_median(r):
    return np.median(r, axis=ax_multi)


def get_ci(r, c):
    return np.percentile(r, c, axis=ax_multi)


def get_above_eil(r, norm=True):
    if norm:
        return np.maximum(r - eil, 0) / eil
    return np.maximum(r - eil, 0)


def get_risk_above_eil(r):
    return np.mean(np.greater_equal(r, eil), axis=rep_axes)


def _smooth(m, window=15, poly=3, lims=None):
    m = savgol_filter(m, window, poly)
    if lims is not None:
        if lims[0] is not None:
            m = np.maximum(m, lims[0])
        if lims[1] is not None:
            m = np.minimum(m, lims[1])
    return m


def plot_population(ax, m_res, percentiles=None, shadow=False, post=''):
    """
    A useful function to graph modelled population outputs with uncertainty bounds.
    """

    color = '#99CC00'
    # Specify axes in output matrix (time, field, stochastic and parametric uncertainty)
    ax_t, ax_field, ax_stoc, ax_param = (0, 1, -1, -2)

    # Plot median prediction
    x_ = np.arange(m_res.shape[ax_t])
    median_format = {'lw': 1, 'linestyle': '--', 'color': '#000000'} if shadow else {'lw': 2, 'color': color}
    ax.plot(x_, np.median(m_res, axis=(ax_stoc, ax_field, ax_param)), **median_format, label='Median' + post)

    # Percentiles to plot
    percentiles = percentiles or [50, 75, 95]
    percentiles.sort()

    # Minimum and maximum of previous percentile
    max_prc_before = min_prc_before = np.median(m_res, axis=(ax_stoc, ax_field, ax_param))

    # For each percentile...
    for n, p in enumerate(percentiles):
        # Maximum and minimum percentiles of data
        max_prc = np.percentile(m_res, 50 + p / 2, axis=(ax_stoc, ax_field, ax_param))
        min_prc = np.percentile(m_res, (100 - p) / 2, axis=(ax_stoc, ax_field, ax_param))

        if shadow:
            ax.plot(x_, max_prc, lw=1, linestyle=':', color='#000000', label='CI {} %'.format(p) + post)
            ax.plot(x_, min_prc, lw=1, linestyle=':', color='#000000')
        else:
            # Calculate % opacity and draw
            max_op = 0.6
            min_op = 0.2
            opacity = (1 - n / (len(percentiles) - 1)) * (max_op - min_op) + min_op

            ax.fill_between(
                x_, max_prc_before, max_prc,
                facecolor=color, alpha=opacity, linewidth=0.5, edgecolor=color, label='CI {} %'.format(p) + post
            )
            ax.fill_between(
                x_, min_prc, min_prc_before,
                facecolor=color, alpha=opacity, linewidth=0.5, edgecolor=color
            )

            # Save minimum and maximum lines for next percentile
            min_prc_before = min_prc
            max_prc_before = max_prc


if __name__ == '__main__':
    base_data = BaseRun.get_data()
    larvae_base = base_data['sum_larvae']
    larvae_345 = np.sum([base_data['O. arenosella juvenil_%i' % i] for i in range(3, 6)], axis=0)
    pupal_base = base_data['O. arenosella pupa']
    pupal_paras_base = base_data['Parasitoide pupa juvenil']
    larval_paras_base = base_data['Parasitoide larvas juvenil']

    if not os.path.isdir(dir_figs):
        os.makedirs(dir_figs)

    # Figure 1
    to_include = {
        'Pesticide except eggs': RunPesticideExcptEggs,
        'General pesticide': RunPesticideGeneral,
        'Pupal parasitoid biocontrol': RunBiocontrolPupa,
    }
    data = {ll: v.get_data() for ll, v in to_include.items()}

    fig = Figure(figsize=(6 * 3, 6))
    FigureCanvasAgg(fig)
    axes = [ax1, ax2, ax3] = fig.subplots(ncols=3)
    ax2.get_shared_y_axes().join(ax2, ax3)

    slc = slice(60, None)
    cut_dmg = 60
    fixed_days = np.array(time_range)
    for run, res in data.items():
        x, res_t = slice_time(res, sl=slc)
        risk = get_risk_above_eil(res_t)

        ax1.plot(x, get_median(risk), label=run)
        ax1.fill_between(x, get_ci(risk, 5), get_ci(risk, 95), alpha=0.25)


        def plot_damage(ax, dmg):
            ax.plot(fixed_days, _smooth(np.median(dmg, axis=rep_axes), window=9, lims=(0, None)))
            ax.fill_between(
                fixed_days,
                _smooth(np.percentile(dmg, 5, axis=rep_axes), window=9, lims=(0, None)),
                _smooth(np.percentile(dmg, 95, axis=rep_axes), window=9, lims=(0, None)),
                alpha=0.25
            )


        above_eil_start = np.sum(get_above_eil(slice_time(res, sl=slice(None, cut_dmg))[1]), axis=ax_time)
        above_eil_end = np.sum(get_above_eil(slice_time(res, sl=slice(cut_dmg, None))[1]), axis=ax_time)

        plot_damage(ax2, above_eil_start)
        plot_damage(ax3, above_eil_end)

    x_base, base_data = slice_time(larvae_base, sl=slc, multi=False)
    ax1.plot(x_base, get_risk_above_eil(base_data), linestyle='dashed', color='#000000', label='Without control')

    base_above_eil = np.median(get_above_eil(larvae_base), axis=rep_axes)
    ax2.plot(fixed_days, np.full(fixed_days.shape, np.sum(base_above_eil[:cut_dmg])), linestyle='dashed',
             color='#000000')
    ax3.plot(fixed_days, np.full(fixed_days.shape, np.sum(base_above_eil[cut_dmg:])), linestyle='dashed',
             color='#000000')

    ax1.set_ylabel('Risk of economic injury', fontsize=16)
    ax1.set_xlabel('Day of simulation', fontsize=16)
    ax1.set_title('Risk of economic injury by day', fontsize=18)

    ax2.set_ylabel('Economic injury (NED)', fontsize=16)
    ax2.set_xlabel('Day of action', fontsize=16)
    ax2.set_title(f'Total economic injury, days 0-{cut_dmg}', fontsize=18)

    ax3.set_xlabel('Day of action', fontsize=16)
    ax3.set_title(f'Total economic injury, days {cut_dmg}-end', fontsize=18)

    fig.suptitle('Efficiency and risk of fixed-day control strategies', fontsize=25)
    fig.subplots_adjust(bottom=0.22, top=0.8, wspace=0.3, left=0.07, right=1 - 0.07)
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=15)

    fig.savefig(f'{dir_figs}/Fig 1.jpeg')

    # Figure 2
    fig = Figure(figsize=(12, 14))
    FigureCanvasAgg(fig)
    axes = [(ax1, ax2), (ax3, ax4), (ax5, ax6)] = fig.subplots(ncols=2, nrows=3)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax3.get_shared_y_axes().join(ax3, ax4)

    to_include = {
        ax1: {
            'Fixed date': RunBiocontrolLarva, 'Economic threshold': DRunBiocontrolLarva, 'Optimised': ORunBiocontrolLarvas3
        },
        ax2: {
            'Fixed date': RunBiocontrolPupa, 'Economic threshold': DRunBiocontrolPupa, 'Optimised': ORunBiocontrolPupa3
        }
    }
    data = {axis: {name: run.get_data() for name, run in include.items()} for axis, include in to_include.items()}
    without_paras_pupa = NoPupalParasto150.get_data('sum_larvae')
    without_paras_larvae = NoLarvalParasto150.get_data('sum_larvae')

    for axis, include in data.items():

        for name, run in include.items():
            x, res_t = slice_time(run, sl=slc)
            risk = get_risk_above_eil(res_t)
            axis.plot(x, get_median(risk), label=name)
            axis.fill_between(x, get_ci(risk, 5), get_ci(risk, 95), alpha=0.25)

    time_slice = slice(25, 200)

    ax3.scatter(larvae_345[time_slice], larval_paras_base[time_slice] / larvae_345[time_slice], alpha=0.01)
    ax4.scatter(pupal_base[time_slice], pupal_paras_base[time_slice] / pupal_base[time_slice], alpha=0.01)

    plot_population(ax5, larvae_base, percentiles=[95], shadow=True, post=' base run')
    plot_population(ax5, without_paras_larvae)
    ax5.annotate(
        'Reintroduced', xy=(150, 2.5e6), xytext=(150, 3.5e6),
        arrowprops=dict(facecolor='black', arrowstyle="->"), ha='center',
    )

    plot_population(ax6, larvae_base, percentiles=[95], shadow=True, post=' base run')
    plot_population(ax6, without_paras_pupa)
    ax6.annotate(
        'Reintroduced', xy=(150, 1.15e7), xytext=(150, 1.6e7),
        arrowprops=dict(facecolor='black', arrowstyle="->"), ha='center',
    )

    ax1.set_title('Larval parasitoid - biocontrol', fontsize=18)
    ax1.set_ylabel('Risk of economic injury', fontsize=16)
    ax1.set_xlabel('Day of simulation', fontsize=16)
    ax1.legend(fontsize=11)

    ax2.set_title('Pupal parasitoid - biocontrol', fontsize=18)
    ax2.set_xlabel('Day of simulation', fontsize=16)
    ax2.set_yticklabels([])

    ax3.set_title('Larval parasitoid - efficiency', fontsize=18)
    ax3.set_ylabel('Parasitism (%)', fontsize=14)
    ax3.set_xlabel('Host population (ha -1)', fontsize=16)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax4.set_title('Pupal parasitoid - efficiency', fontsize=18)
    ax4.set_xlabel('Host population (ha -1)', fontsize=16)
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax4.set_yticklabels([])

    ax5.set_title('Without larval parasitoid', fontsize=18)
    ax5.set_ylabel('O. arenosella larvae (ha -1)', fontsize=14)
    ax5.set_xlabel('Day of simulation', fontsize=16)
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax5.legend(fontsize=11)

    ax6.set_title('Without pupal parasitoid', fontsize=18)
    ax6.set_xlabel('Day of simulation', fontsize=16)
    ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.suptitle('Efficiency of biocontrol strategies', fontsize=25)
    fig.subplots_adjust(wspace=0.15, hspace=0.4)

    fig.savefig(f'{dir_figs}/Fig 2.jpeg')