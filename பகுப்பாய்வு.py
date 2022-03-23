import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from tikon.móds.rae.utils import EJE_ETAPA
from tikon.utils import EJE_TIEMPO, EJE_ESTOC, EJE_PARÁMS, EJE_PARC, proc_líms
from எண்ணிக்கை import உரைக்கு as உ

from model import exper_A
from runs import eil, RunPesticideExcptEggs, RunPesticideGeneral, RunSimplePesticideAdult, RunBiocontrolPupa, DRunBiocontrolPupa, \
    time_range, DRunBiocontrolLarva, RunBiocontrolLarva, ORunBiocontrolPupa3, ORunBiocontrolLarvas3, BaseRun, \
    NoPupalParasto150, NoLarvalParasto150, RunSimpleBiocontrolLarva, RunSimplePesticideGeneral, RunSimpleBiocontrolPupa, RunSimplePesticideExceptEggs, start_date
from எழுத்துரு import எழுத்துரு

"""
This is the code used to run all analyses and generate all figures present in the article.
"""

dir_figs = 'out/figs'
parallel = True

dims_reps = [EJE_ESTOC, EJE_PARÁMS]


def தமிழ்_அச்சுகள்(_அச்சு):
    _அச்சு.set_xticklabels(_அச்சு.get_xticks(), fontproperties=எழுத்துரு)
    _அச்சு.set_yticklabels(_அச்சு.get_yticks(), fontproperties=எழுத்துரு)
    _அச்சு.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: உ(int(x), 'தமிழ்'))
    )
    _அச்சு.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: உ(int(x), 'தமிழ்'))
    )


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

    ax.plot(t, pobs, color='#000000', marker='o', markersize=3, label='அவதானிக்கப்பட்டது')


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
    ax.plot(x_, res_.median(dim=dims_reps), **median_format, label='இடைநிலையளவு' + post)

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
            ax.plot(x_, max_prc, lw=1, linestyle=':', color='#000000',
                    label='{} % ந. இடை.'.format(உ(p, "தமிழ்")) + post)
            ax.plot(x_, min_prc, lw=1, linestyle=':', color='#000000')
        else:
            # Calculate % opacity and draw
            max_op = 0.6
            min_op = 0.2
            opacity = (1 - n / (len(quantiles) - 1)) * (max_op - min_op) + min_op

            ax.fill_between(
                x_, max_prc_before, max_prc,
                facecolor=color, alpha=opacity, linewidth=0.5, edgecolor=color,
                label='{} % ந. இடை.'.format(உ(p, "தமிழ்")) + post
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
    base_data = BaseRun.get_data().squeeze(EJE_PARC)  # We only have 1 field anyways
    larvae_base = base_data.loc[{EJE_ETAPA: 'sum larvae'}]
    larvae_345_base = base_data.loc[{EJE_ETAPA: ['O. arenosella : juvenil %i' % i for i in range(3, 6)]}].sum(EJE_ETAPA)
    pupae_base = base_data.loc[{EJE_ETAPA: 'O. arenosella : pupa'}]
    pupal_paras_base = base_data.loc[{EJE_ETAPA: 'Parasitoide pupa : juvenil'}]
    larval_paras_base = base_data.loc[{EJE_ETAPA: 'Parasitoide larvas : juvenil'}]

    if not os.path.isdir(dir_figs):
        os.makedirs(dir_figs)

    # Figure 2
    fig = plt.figure(figsize=(12, 10))
    (ax1, ax2), (ax3, ax4) = fig.subplots(ncols=2, nrows=2, sharex='all')

    plot_population(ax1, larvae_base, t=slice(0, 300))
    plot_population(ax2, pupae_base, t=slice(0, 300))
    plot_population(ax3, larval_paras_base, t=slice(0, 300))
    plot_population(ax4, pupal_paras_base, t=slice(0, 300))

    plot_pobs(ax1, stages=[f'O. arenosella : juvenil {i}' for i in range(1, 6)])
    plot_pobs(ax2, stages='O. arenosella : pupa')
    plot_pobs(ax3, stages='Parasitoide larvas : juvenil')
    plot_pobs(ax4, stages='Parasitoide pupa : juvenil')

    ax1.set_title('தென்னைக் கருந்தலைப்புழு குடம்பி', fontsize=15, fontproperties=எழுத்துரு)
    ax2.set_title('தென்னைக் கருந்தலைப்புழு கூட்டுப்புழு', fontsize=15, fontproperties=எழுத்துரு)
    ax3.set_title('குடம்பி ஒட்டுண்ணியின் குடம்பி', fontsize=15, fontproperties=எழுத்துரு)
    ax4.set_title('கூட்டுப்புழு ஒட்டுண்ணியி்ன் குடம்பி', fontsize=15, fontproperties=எழுத்துரு)

    ax3.set_xlabel('நாட்கள்', fontsize=15, fontproperties=எழுத்துரு)
    ax4.set_xlabel('நாட்கள்', fontsize=15, fontproperties=எழுத்துரு)

    for அச்சு in [ax1, ax2, ax3, ax4]:
        தமிழ்_அச்சுகள்(அச்சு)

    fig.autofmt_xdate(rotation=25)
    fig.legend(
        *ax1.get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=15, prop=எழுத்துரு
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.20)
    fig.savefig(f'{dir_figs}/உருப்படம் - ௧.jpeg')

    # Figure 3
    to_include = {
        'பூச்சிக்கொல்லி (முட்டை தவிர)': RunSimplePesticideExceptEggs,
        'பொதுவான பூச்சிக்கொல்லி': RunSimplePesticideGeneral,
        'பெரியவர்கள் பூச்சிக்கொல்லி': RunSimplePesticideAdult,
        'கூட்டுப்புழு ஒட்டுண்ணியால் உயிர் கட்டு்ப்பாடு': RunSimpleBiocontrolPupa,
        'குடம்பி ஒட்டுண்ணியால் உயிர் கட்டு்ப்பாடு': RunSimpleBiocontrolLarva,
    }
    data = {
        ll: v.get_data().squeeze(EJE_PARC).stack(répli=["paráms", "estoc"]) for ll, v in to_include.items()
    }

    fig = plt.figure(figsize=(6 * 3, 6 * 3))
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = fig.subplots(ncols=2, nrows=3)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    slc12 = slice(60, None)
    cut_dmg = 60
    fixed_days = np.array(time_range)

    pops = {
        'மூலம்': {
            'larvae': larvae_base,
            'larvae_345': larvae_345_base,
            'pupae': pupae_base,
            'pupal_paras': pupal_paras_base,
            'larval_paras': larval_paras_base
        }
    }
    for run, res in data.items():
        pops[run] = {
            'larvae': res.loc[{EJE_ETAPA: 'sum larvae'}],
            'larvae_345': res.loc[{EJE_ETAPA: ['O. arenosella : juvenil %i' % i for i in range(3, 6)]}].sum(EJE_ETAPA),
            'pupae': res.loc[{EJE_ETAPA: 'O. arenosella : pupa'}],
            'pupal_paras': res.loc[{EJE_ETAPA: 'Parasitoide pupa : adulto'}],
            'larval_paras': res.loc[{EJE_ETAPA: 'Parasitoide larvas : adulto'}]
        }

    for ax, run in zip(axes, pops):
#         plot_population(ax, pops[run]["larvae"])

        ax.scatter(pops[run]['larval_paras'], pops[run]['larvae'],  alpha=0.01)
        ax.set_title(run, fontsize=15, fontproperties=எழுத்துரு)

    for அச்சு in axes:
        தமிழ்_அச்சுகள்(அச்சு)

    ax1.set_xlabel('Jour de simulation', fontsize=16)

    ax2.set_xlabel('Jour d\'action', fontsize=16)

    ax3.set_xlabel('Jour d\'action', fontsize=16)

    fig.suptitle('Efficacité et risque des stratégies de contrôle à date fixe', fontsize=25)
    fig.subplots_adjust(wspace=0.15, hspace=0.4)
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=15, prop=எழுத்துரு)

    fig.savefig(f'{dir_figs}/உருப்படம் - ௨.jpeg')

    # Figure 4
    fig = Figure()
    FigureCanvasAgg(fig)
    axes = fig.subplots()
    fit_larvae, fit_pupa = ORunBiocontrolLarvas3.get_fit(), ORunBiocontrolPupa3.get_fit()
    axes.plot(-fit_larvae.cummax(), label='Parasitoïde larves')
    axes.plot(-fit_pupa.cummax(), label='Parasitoïde pupe')

    axes.set_xlabel('Itération', fontsize=16)
    axes.set_ylabel('Meilleure valeure à date', fontsize=16)
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
            'Date fixe': RunBiocontrolLarva, 'Seuil économique': DRunBiocontrolLarva,
            'Optimisé': ORunBiocontrolLarvas3
        },
        ax2: {
            'Date fixe': RunBiocontrolPupa, 'Seuil économique': DRunBiocontrolPupa, 'Optimisé': ORunBiocontrolPupa3
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
        larvae_345_base[{EJE_TIEMPO: slc34}],
        larval_paras_base[{EJE_TIEMPO: slc34}] / larvae_345_base[{EJE_TIEMPO: slc34}],
        alpha=0.01
    )
    ax4.scatter(
        pupae_base[{EJE_TIEMPO: slc34}], pupal_paras_base[{EJE_TIEMPO: slc34}] / pupae_base[{EJE_TIEMPO: slc34}],
        alpha=0.01
    )

    plot_population(ax5, larvae_base, quantiles=[.95], shadow=True, post=' référence')
    plot_population(ax5, without_paras_larvae)
    ax5.annotate(
        'Réintroduction', xy=(150, 2.5e6), xytext=(150, 3.5e6),
        arrowprops=dict(facecolor='black', arrowstyle="->"), ha='center',
    )

    plot_population(ax6, larvae_base, quantiles=[.95], shadow=True, post=' référence')
    plot_population(ax6, without_paras_pupa)
    ax6.annotate(
        'Réintroduction', xy=(150, 1.15e7), xytext=(150, 1.6e7),
        arrowprops=dict(facecolor='black', arrowstyle="->"), ha='center',
    )

    ax1.set_title('Parasitoïde larve - biocontrôle', fontsize=18)
    ax1.set_ylabel('Risque dommage économique', fontsize=16)
    ax1.set_xlabel('Jour de simulation', fontsize=16)
    ax1.legend(fontsize=11)

    ax2.set_title('Parasitoïde pupe - biocontrôle', fontsize=18)
    ax2.set_xlabel('Jour de simulation', fontsize=16)
    ax2.set_yticklabels([])

    ax3.set_title('Parasitoïde larves - efficacité', fontsize=18)
    ax3.set_ylabel('Parasitisme (%)', fontsize=14)
    ax3.set_xlabel('Population de l\'hôte (par ha)', fontsize=16)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax4.set_title('Parasitoïde pupe - efficacité', fontsize=18)
    ax4.set_xlabel('Population de l\'hôte (par ha)', fontsize=16)
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax4.set_yticklabels([])

    ax5.set_title('Sans parasitoïde larves', fontsize=18)
    ax5.set_ylabel('O. arenosella larves (par ha)', fontsize=14)
    ax5.set_xlabel('Jour de simulation', fontsize=16)
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax5.legend(fontsize=11)

    ax6.set_title('Sans parasitoïde pupe', fontsize=18)
    ax6.set_xlabel('Jour de simulation', fontsize=16)
    ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.suptitle('Efficacité des stratégies de biocontrôle', fontsize=25)
    fig.subplots_adjust(wspace=0.15, hspace=0.4)

    fig.savefig(f'{dir_figs}/உருப்படம் 5.jpeg')
