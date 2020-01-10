import os
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

from model import exper_A, web
from runs import eil, RunPesticideExcptEggs, RunPesticideGeneral, RunBiocontrolPupa, DRunBiocontrolPupa, \
    time_range, DRunBiocontrolLarva, RunBiocontrolLarva, ORunBiocontrolPupa3, ORunBiocontrolLarvas3, BaseRun, \
    NoPupalParasto150, NoLarvalParasto150, start_date, reps
from tikon.central import Modelo
from tikon.central.calibs import EspecCalibsCorrida
from tikon.datos.proc import r2, ens
from tikon.móds.rae.utils import EJE_ETAPA
from tikon.utils import EJE_TIEMPO, EJE_ESTOC, EJE_PARÁMS, EJE_PARC, proc_líms

"""
This is the code used to run all analyses and generate all figures present in the article.
"""

dir_figs = 'out/figs'
parallel = True

dims_reps = [EJE_ESTOC, EJE_PARÁMS]


def get_above_eil(r, norm=True):
    if norm:
        return np.maximum(r - eil, 0) / eil
    return np.maximum(r - eil, 0)


def get_risk_above_eil(r):
    return np.greater_equal(r, eil).mean(dim=dims_reps)


def _smooth(m, window=15, poly=3, lims=None):
    m = savgol_filter(m, window, poly)
    lims = proc_líms(lims)
    return np.maximum(np.minimum(m, lims[1]), lims[0])


def plot_pobs(ax, stages):
    obs = exper_A.datos.obt_obs('red', var='Pobs')[0].datos.copy()
    obs.coords[EJE_ETAPA] = [str(y) for y in obs.coords[EJE_ETAPA].values]

    t = obs[EJE_TIEMPO]
    pobs = obs.loc[{EJE_ETAPA: stages}].squeeze(EJE_PARC)
    if isinstance(stages, list):
        pobs = pobs.sum(dim=EJE_ETAPA)

    ax.plot(t, pobs, color='#000000', marker='o', markersize=3, label='Observed')


def plot_population(ax, res_, quantiles=None, shadow=False, post='', t=None):
    """
    A useful function to graph modelled population outputs with uncertainty bounds.
    """

    color = '#99CC00'

    if t:
        res_ = res_[{EJE_TIEMPO: t}]

    # Plot median prediction
    x_ = _get_days(res_)
    median_format = {'lw': 1, 'linestyle': '--', 'color': '#000000'} if shadow else {'lw': 2, 'color': color}
    ax.plot(x_, res_.median(dim=dims_reps), **median_format, label='Median' + post)

    # Quantile to plot
    quantiles = quantiles or [.50, .75, .95]

    # Minimum and maximum of previous quantile
    max_prc_before = min_prc_before = res_.median(dim=dims_reps)

    # For each quantile...
    for n, q in enumerate(quantiles):

        # Maximum and minimum percentiles of data
        max_prc = res_.quantile(0.50 + q / 2, dim=dims_reps)
        min_prc = res_.quantile((1 - q) / 2, dim=dims_reps)

        p = int(q * 100)
        if shadow:
            ax.plot(x_, max_prc, lw=1, linestyle=':', color='#000000', label='CI {} %'.format(p) + post)
            ax.plot(x_, min_prc, lw=1, linestyle=':', color='#000000')
        else:
            # Calculate % opacity and draw
            max_op = 0.6
            min_op = 0.2
            opacity = (1 - n / (len(quantiles) - 1)) * (max_op - min_op) + min_op

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


def _get_days(m):
    return pd.Series(pd.to_datetime(m[EJE_TIEMPO].values) - pd.to_datetime(start_date)).dt.days.values


if __name__ == '__main__':

    model = Modelo(web)
    res = model.simular('', reps=reps, exper=exper_A, calibs=EspecCalibsCorrida(aprioris=False))
    pprint(res.validar(proc=[ens, r2]).a_dic())

    base_data = BaseRun.get_data().squeeze(EJE_PARC)  # We only have 1 field anyways
    larvae_base = base_data.loc[{EJE_ETAPA: 'sum larvae'}]
    larvae_345 = base_data.loc[{EJE_ETAPA: ['O. arenosella : juvenil %i' % i for i in range(3, 6)]}].sum(EJE_ETAPA)
    pupae_base = base_data.loc[{EJE_ETAPA: 'O. arenosella : pupa'}]
    pupal_paras_base = base_data.loc[{EJE_ETAPA: 'Parasitoide pupa : juvenil'}]
    larval_paras_base = base_data.loc[{EJE_ETAPA: 'Parasitoide larvas : juvenil'}]

    if not os.path.isdir(dir_figs):
        os.makedirs(dir_figs)

    # Figure 2
    fig = Figure(figsize=(12, 10))
    FigureCanvasAgg(fig)
    (ax1, ax2), (ax3, ax4) = fig.subplots(ncols=2, nrows=2, sharex='all')

    plot_population(ax1, larvae_base, t=slice(0, 300))
    plot_population(ax2, pupae_base, t=slice(0, 300))
    plot_population(ax3, larval_paras_base, t=slice(0, 300))
    plot_population(ax4, pupal_paras_base, t=slice(0, 300))

    plot_pobs(ax1, stages=[f'O. arenosella : juvenil {i}' for i in range(1, 6)])
    plot_pobs(ax2, stages='O. arenosella : pupa')
    plot_pobs(ax3, stages='Parasitoide larvas : juvenil')
    plot_pobs(ax4, stages='Parasitoide pupa : juvenil')

    ax1.set_title('O. arenosella larvae', fontsize=20)
    ax2.set_title('O. arenosella pupae', fontsize=20)
    ax3.set_title('Juvenile larval parasitoid', fontsize=20)
    ax4.set_title('Juvenile pupal parasitoid', fontsize=20)

    ax1.set_ylabel('Population (ha-1)', fontsize=20)
    ax3.set_ylabel('Population (ha-1)', fontsize=20)
    ax3.set_xlabel('Days', fontsize=20)
    ax4.set_xlabel('Days', fontsize=20)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.autofmt_xdate()
    fig.legend(
        *ax1.get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=20,
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.20)
    fig.savefig(f'{dir_figs}/Fig 2.jpeg')

    # Figure 3
    to_include = {
        'Pesticide except eggs': RunPesticideExcptEggs,
        'General pesticide': RunPesticideGeneral,
        'Pupal parasitoid biocontrol': RunBiocontrolPupa,
    }
    data = {ll: v.get_data(parallel=parallel).squeeze(EJE_PARC) for ll, v in to_include.items()}

    fig = Figure(figsize=(6 * 3, 6))
    FigureCanvasAgg(fig)
    axes = [ax1, ax2, ax3] = fig.subplots(ncols=3)

    slc12 = slice(60, None)
    cut_dmg = 60
    fixed_days = np.array(time_range)

    for run, res in data.items():
        res_t = res[{EJE_TIEMPO: slc12}]
        risk = get_risk_above_eil(res_t)

        x = _get_days(res_t)
        ax1.plot(x, risk.median(dim='multi'), label=run)
        ax1.fill_between(x, risk.quantile(0.05, dim='multi'), risk.quantile(0.95, dim='multi'), alpha=0.25)


        def plot_damage(ax, dmg):
            ax.plot(fixed_days, _smooth(dmg.median(dim=dims_reps), window=9, lims=(0, None)))
            ax.fill_between(
                fixed_days,
                _smooth(dmg.quantile(.05, dim=dims_reps), window=9, lims=(0, None)),
                _smooth(dmg.quantile(.95, dim=dims_reps), window=9, lims=(0, None)),
                alpha=0.25
            )


        above_eil_start = get_above_eil(res[{EJE_TIEMPO: slice(None, cut_dmg)}]).sum(dim=EJE_TIEMPO)
        above_eil_end = get_above_eil(res[{EJE_TIEMPO: slice(cut_dmg, None)}]).sum(dim=EJE_TIEMPO)

        plot_damage(ax2, above_eil_start)
        plot_damage(ax3, above_eil_end)

    base_data = larvae_base[{EJE_TIEMPO: slc12}]
    x_base = _get_days(base_data)
    ax1.plot(x_base, get_risk_above_eil(base_data), linestyle='dashed', color='#000000', label='Without control')

    base_above_eil = get_above_eil(larvae_base).median(dim=dims_reps)
    ax2.plot(
        fixed_days, np.full(fixed_days.shape, base_above_eil[{EJE_TIEMPO: slice(None, cut_dmg)}].sum()),
        linestyle='dashed', color='#000000'
    )
    ax3.plot(
        fixed_days, np.full(fixed_days.shape, base_above_eil[{EJE_TIEMPO: slice(cut_dmg, None)}].sum()),
        linestyle='dashed',
        color='#000000'
    )

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

    fig.savefig(f'{dir_figs}/Fig 3.jpeg')

    # Figure 4
    fig = Figure()
    FigureCanvasAgg(fig)
    axes = fig.subplots()
    fit_larvae, fit_pupa = ORunBiocontrolLarvas3.get_fit(), ORunBiocontrolPupa3.get_fit()
    axes.plot(-fit_larvae.cummax(), label='Larval parasitoid')
    axes.plot(-fit_pupa.cummax(), label='Pupal parasitoid')

    axes.set_xlabel('Iteration', fontsize=16)
    axes.set_ylabel('Best function value', fontsize=16)
    fig.legend(loc='lower center', ncol=2, fontsize=12)

    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f'{dir_figs}/Fig 4.jpeg')

    # Figure 5
    fig = Figure(figsize=(12, 14))
    FigureCanvasAgg(fig)
    axes = [(ax1, ax2), (ax3, ax4), (ax5, ax6)] = fig.subplots(ncols=2, nrows=3)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax3.get_shared_y_axes().join(ax3, ax4)

    to_include = {
        ax1: {
            'Fixed date': RunBiocontrolLarva, 'Economic threshold': DRunBiocontrolLarva,
            'Optimised': ORunBiocontrolLarvas3
        },
        ax2: {
            'Fixed date': RunBiocontrolPupa, 'Economic threshold': DRunBiocontrolPupa, 'Optimised': ORunBiocontrolPupa3
        }
    }
    data = {
        axis: {name: run.get_data(parallel=parallel).squeeze(EJE_PARC) for name, run in include.items()}
        for axis, include in to_include.items()
    }
    without_paras_pupa = NoPupalParasto150.get_data('sum larvae').squeeze(EJE_PARC)
    without_paras_larvae = NoLarvalParasto150.get_data('sum larvae').squeeze(EJE_PARC)

    for axis, include in data.items():

        for name, run in include.items():
            res_t = run[{EJE_TIEMPO: slc12}]
            x = _get_days(res_t)
            risk = get_risk_above_eil(res_t)
            axis.plot(x, risk.median(dim='multi'), label=name)
            axis.fill_between(x, risk.quantile(0.05, dim='multi'), risk.quantile(0.95, dim='multi'), alpha=0.25)

    slc34 = slice(25, 200)

    ax3.scatter(
        larvae_345[{EJE_TIEMPO: slc34}], larval_paras_base[{EJE_TIEMPO: slc34}] / larvae_345[{EJE_TIEMPO: slc34}],
        alpha=0.01
    )
    ax4.scatter(
        pupae_base[{EJE_TIEMPO: slc34}], pupal_paras_base[{EJE_TIEMPO: slc34}] / pupae_base[{EJE_TIEMPO: slc34}],
        alpha=0.01
    )

    plot_population(ax5, larvae_base, quantiles=[.95], shadow=True, post=' base run')
    plot_population(ax5, without_paras_larvae)
    ax5.annotate(
        'Reintroduced', xy=(150, 2.5e6), xytext=(150, 3.5e6),
        arrowprops=dict(facecolor='black', arrowstyle="->"), ha='center',
    )

    plot_population(ax6, larvae_base, quantiles=[.95], shadow=True, post=' base run')
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
    ax3.set_xlabel('Host population (per ha)', fontsize=16)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax4.set_title('Pupal parasitoid - efficiency', fontsize=18)
    ax4.set_xlabel('Host population (per ha)', fontsize=16)
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax4.set_yticklabels([])

    ax5.set_title('Without larval parasitoid', fontsize=18)
    ax5.set_ylabel('O. arenosella larvae (per ha)', fontsize=14)
    ax5.set_xlabel('Day of simulation', fontsize=16)
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax5.legend(fontsize=11)

    ax6.set_title('Without pupal parasitoid', fontsize=18)
    ax6.set_xlabel('Day of simulation', fontsize=16)
    ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.suptitle('Efficiency of biocontrol strategies', fontsize=25)
    fig.subplots_adjust(wspace=0.15, hspace=0.4)

    fig.savefig(f'{dir_figs}/Fig 5.jpeg')
