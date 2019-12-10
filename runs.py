import os
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd
import spotpy
from tikon.central import Modelo
from tikon.móds.rae.manejo import AgregarPob, MultPob, Acción
from tikon.móds.manejo.conds import CondDía, SuperiorOIgual, CondCadaDía, Inferior
from tikon.móds.rae.manejo import CondPoblación
from tikon.móds.manejo import Manejo, Regla
from tikon.utils import guardar_json, leer_json

from model import web, exper_A, Oarenosella, Larval_paras, Pupal_paras

"""
This code specifies all simulation runs used in the article. It does not actually run any simulations itself, but
provides run specifications for the `analysis.py` file, where they are then run on demand (and so should not be run
directly).
All simulation settings are specified here.
"""

# Simulation settings
start_date = '1982-04-01'
final_day = 400
n_rep_param = 50
n_rep_estoc = 5

n_iter_opt = 500
n_keep_opt = 10

opt_day_range = (25, 150)
verbose = True
out_dir = 'out'

# Constants
parasitoid_dose = 600000
dynamic_paras_dose = parasitoid_dose / 3

eil = 655757.1429 * 0.5
survival = 0.05  # Survival rate from pesticides

# Load calibrations
web.cargar_calib('Site A calibs/red')
exper_A.cargar_calib('Site A calibs')

# Get all life stages
stages = [s for o in web for s in o]

# "Ghosts" are Tiko'n's representation of parasitised insects that will mature into parasitoids
ghosts_larvae = ['Parasitoide larvas juvenil en O. arenosella juvenil_{}'.format(i) for i in range(3, 6)]
ghosts_pupa = ['Parasitoide pupa juvenil en O. arenosella pupa']


def get_larvae(n):
    """
    Returns all O. arenosella larval stages in the model.

    Parameters
    ----------
    n: list or int
        A list of the larval stage indices of interest (1 - 5).

    Returns
    -------
    list
        Larval stage objects or names, including parasitised (ghost) stages.
    """

    if isinstance(n, int):
        n = [n]
    stgs = [x for x in stages if x.org is Oarenosella and any(x.nombre.endswith('juvenil_%i' % i) for i in n)]
    stgs += ['Parasitoide larvas juvenil en O. arenosella juvenil_{}'.format(i) for i in n]
    return stgs


# Quick reference to specific life stages
larvae = [s for s in stages if s.org is Oarenosella and 'juvenil' in s.nombre] + ghosts_larvae
adults = [s for s in stages if s.nombre == 'adulto']
pupa = [s for s in stages if s.org is Oarenosella and s.nombre == 'pupa'] + ghosts_pupa
not_eggs = [s for s in stages if s.nombre != 'huevo'] + ghosts_larvae + ghosts_pupa
not_sedent = [s for s in stages if s.nombre != 'pupa' and s.nombre != 'huevo'] + ghosts_larvae
all_stages = stages + ghosts_larvae + ghosts_pupa


class SingleRun(object):
    """
    Class to set up a single model run.
    """

    def __init__(self, name, mgmt, all_vars=False):
        """
        Initialised the run specification.

        Parameters
        ----------
        name: str
            Name of the run (used to write output).
        mgmt: Manejo
            Management object to apply to the run.
        all_vars: bool
            Whether to include all variables in the output or not.
        """

        self.mgmt = mgmt
        self.name = name
        self.all_vars = all_vars

    def get_data(self, var=None):
        """
        Obtains output data from the run. If the run has not yet been simulated, will also run the simulation.
        Otherwise, it will simply read from the previous simulation execution's output file on disk.

        Parameters
        ----------
        var: str
            Output variable of interest (optional).
        Returns
        -------
        dict | np.ndarray
            Dictionnary of output variables, or numpy array of output variable if `var` was specified.
        """

        filename = self._get_output_filename()
        # Execute run if output file is not found on disk.
        if not os.path.isfile(filename):
            run((filename, self.mgmt, self.all_vars))

        # Return results
        if var is None:
            return leer_json(filename)
        return leer_json(filename)[var]

    def _get_output_filename(self):
        """
        Returns the expected output filename for this run.

        Returns
        -------
        str
        """

        return f'{out_dir}/{self.name}.json'


class MultiRun(object):
    """
    Object to represent runs that have repetitions (e.g., date or otherwise).
    """

    def __init__(self, name, range_):
        """
        Initialises the run.

        Parameters
        ----------
        name: str
            Base name of the run.
        range_: iterable
            Range of values from which to generate multiple runs.
        """

        self.range_ = range_
        self.name = name

    def get_data(self, parallel=True, var='sum_larvae'):
        """
        Obtain data from all runs. Any runs whose output files are not found on disk will be re-run.

        Parameters
        ----------
        parallel: bool
            Whether to run in parallel on multiple computer cores or not.
        var: str
            Output variable of interest.

        Returns
        -------
        np.ndarray
            Array of output variable. Axis 0 is along `self.range_`.
        """

        self.run(parallel=parallel)
        return np.array(
            [leer_json(self._get_output_filename(i))[var] for i in self.range_]
        )

    def run(self, parallel=True):
        """
        Executes all missing runs.

        Parameters
        ----------
        parallel: bool
            Whether to run in parallel or not.
        """

        # Only execute runs that do not have output files already saved to disk.
        to_run = {}
        for i in self.range_:
            filename = self._get_output_filename(i)
            if not os.path.isfile(filename):
                to_run[filename] = self._get_mgmt(i)

        # Run all missing runs, either in parallel or else sequentially.
        if parallel:
            with Pool() as p:
                p.map(run, to_run.items())
        else:
            for f, s in to_run.items():
                run(f, s)

    def _get_output_filename(self, i):
        """
        Gets the output file name for a specific run.

        Parameters
        ----------
        i: int
            Index of the run.

        Returns
        -------
        str
        """

        return f'{out_dir}/{self.name}/{i}.json'

    def _get_mgmt(self, i):
        """
        Must return the corresponding management object for a particular individual run.

        Parameters
        ----------
        i: int
            The run index.

        Returns
        -------
        Manejo
        """

        raise NotImplementedError


class DateRun(MultiRun):
    """
    A `MultiRun` where multiple runs represent the same management action applied on different dates.
    """
    def __init__(self, name, range_, action):
        """
        Initialises the runs.

        Parameters
        ----------
        name: str
            The base name for the runs.
        range_: iterable
            The range of days on which to apply the management action.
        action: Acción
            The action to take.
        """

        self.action = action
        super().__init__(name, range_=range_)

    def _get_mgmt(self, i):
        return Manejo(Regla(CondTiempo(i), self.action))


class DynamicRun(MultiRun):
    """
    A `MultiRun` where multiple runs represent runs with the same action triggered by different population levels.
    """

    def __init__(self, name, range_, trigger, action):
        """
        Initialise the runs.

        Parameters
        ----------
        name: str
            The run base name.
        range_: iterable
            The range of population level triggers to test.
        trigger: str | list | Etapa
            The insect stage(s) whose populations will trigger the action.
        action: Acción
            The management action that will be triggered.
        """
        self.trigger = trigger
        self.action = action
        super().__init__(name, range_=range_)

    def _get_mgmt(self, i):
        # Set condition where population is over or equal to the threshold i, with 30 day minimum wait
        # before re-triggering.
        condition = CondPoblación(self.trigger, SuperiorOIgual(i), espera=30)
        return Manejo(Regla(condition, self.action))


class OptimisedRun(MultiRun):
    def __init__(self, name, n_pupa, n_larva, dose_total):
        self.name = name
        self.n_pupa = n_pupa
        self.n_larva = n_larva
        self.dose_total = dose_total
        super().__init__(name=name, range_=range(n_keep_opt))

    def optimise(self):
        filename = f'{out_dir}/{self.name}/best days.json'
        if os.path.isfile(filename):
            return pd.read_json(filename)

        sampler = spotpy.algorithms.dds(
            _SpotPyMod(lambda x: run((None, self._get_mgmt(x))), self.n_pupa + self.n_larva),
            dbformat='ram', parallel='mpc', save_sim=False, alt_objfun=None
        )
        sampler.sample(n_iter_opt)
        data = sampler.getdata()
        spotpy_out = spotpy.analyser.get_parameters(
            data[np.argpartition(data['like1'], -n_keep_opt)[-n_keep_opt:]]
        )

        pd_out = pd.DataFrame(np.array([list(x) for x in spotpy_out]))

        dir_file = os.path.split(filename)[0]
        if not os.path.isdir(dir_file):
            os.makedirs(dir_file)

        pd_out.to_json(filename)

        return pd_out

    def run(self, parallel=True):
        best_days = self.optimise().values

        to_run = {}
        for i, days in enumerate(best_days):
            filename = self._get_output_filename(i)
            if not os.path.isfile(filename):
                to_run[filename] = self._get_mgmt(days)

        if parallel:
            with Pool() as p:
                p.map(run, to_run.items())
        else:
            for f, s in to_run.items():
                run(f, s)

    def _get_mgmt(self, days):
        n_days = self.n_larva + self.n_pupa
        assert n_days == len(days)
        dose = self.dose_total / n_days
        actions = [
            AgregarPob(Pupal_paras['adulto'] if (t < self.n_pupa) else Larval_paras['adulto'], dose) for t in
            range(n_days)
        ]
        return Manejo([Regla(CondTiempo(int(t)), a) for a, t in zip(actions, days)])


class _SpotPyMod(object):
    """
    Special class for SpotPy integration. See SpotPy documentation for details.
    """
    def __init__(self, run_f, n_days):
        """
        Initialise the class.

        Parameters
        ----------
        run_f: callable
            The function to call to execute a simulation run.
        n_days: int
            The number of control action days to optimise.
        """

        low, high = opt_day_range
        # Set all parameters as uniform distributions
        self.params = [spotpy.parameter.Uniform(str(i), low=low, high=high, as_int=True) for i in range(n_days)]
        self.run_f = run_f

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, x):
        return self._calc_above_eil(self.run_f(x))

    @staticmethod
    def evaluation():
        return [0]  # Not needed; kept for compatibility with SpotPy

    @staticmethod
    def objectivefunction(simulation, evaluation):
        # Note: `evaluation` parameter not used but kept for SpotPy compatibility

        return -np.log(simulation + 1)  # We want to minimise insect-days above the eil

    @staticmethod
    def _calc_above_eil(result):
        return np.sum(np.maximum(0, result['sum_larvae'] - eil)) / eil


def run(*args):
    arg = args[0]
    if len(arg) == 2:
        filename, mgmt = arg
        all_ = False
    else:
        filename, mgmt, all_ = arg

    web_copy = deepcopy(web)
    exp_copy = deepcopy(exper_A)

    if verbose and filename:
        print(f'Running {filename}')

    simul = Simulador([web_copy, mgmt])
    res = simul.simular(final_day, n_rep_parám=n_rep_param, n_rep_estoc=n_rep_estoc, exper=exp_copy)

    d_res_final = process_results(res, all_=all_)
    if filename is not None:
        guardar_json(d_res_final, archivo=filename)

    return d_res_final


def process_results(res, all_=False):
    d_res = {}
    for v in res['red']:
        if v.matr_t is not None:
            d_res[str(v)] = {}
            for e in v.matr_t.dims._coords['etapa']:
                d_res[str(v)][str(e)] = v.matr_t.obt_valor(índs={'etapa': e}).tolist()

    d_res_final = {
        'sum_larvae': np.sum([d_res['Pobs']['O. arenosella juvenil_%i' % i] for i in range(1, 6)], axis=0)
    }
    if all_:
        d_res_final.update(d_res['Pobs'])
    return d_res_final


# Simple runs
BaseRun = SingleRun('no control', mgmt=Manejo(), all_vars=True)
NoPupalParas = SingleRun(
    'without pupal paras', mgmt=Manejo(Regla(CondCada(1), MultPob('Parasitoide pupa adulto', 0)))
)
NoLarvalParas = SingleRun(
    'without larval paras', mgmt=Manejo(Regla(CondCada(1), MultPob('Parasitoide larvas adulto', 0)))
)
NoPupalParasto150 = SingleRun(
    'without pupal paras to 150',
    mgmt=Manejo([
        Regla(CondTiempo(150, Inferior), MultPob('Parasitoide pupa adulto', 0)),
        Regla(CondTiempo(150), AgregarPob('Parasitoide pupa adulto', 100000))
    ])
)
NoLarvalParasto150 = SingleRun(
    'without larval paras to 150',
    mgmt=Manejo([
        Regla(CondTiempo(150, Inferior), MultPob('Parasitoide larvas adulto', 0)),
        Regla(CondTiempo(150), AgregarPob('Parasitoide larvas adulto', 100000))
    ])
)

# Time range for fixed date actions
time_range = range(1, 61, 2)

RunPesticideAdults = DateRun('fd pstcd expt adult', time_range, action=[MultPob(s, survival) for s in adults])
RunPesticideExcptEggs = DateRun('fd pstcd expt eggs', time_range, action=[MultPob(s, survival) for s in not_eggs])
RunPesticideExcptSedent = DateRun('fd pstcd expt sedent', time_range, action=[MultPob(s, survival) for s in not_sedent])
RunPesticideGeneral = DateRun('fd pstcd general', time_range, action=[MultPob(s, survival) for s in all_stages])
RunBiocontrolPupa = DateRun('fd biocntrl pupa', time_range, action=AgregarPob(Pupal_paras['adulto'], parasitoid_dose))
RunBiocontrolLarva = DateRun(
    'fd biocntrl larva', time_range, action=AgregarPob(Larval_paras['adulto'], parasitoid_dose)
)

# Threshold range for dynamic actions
n_thresh = 20
threshold_range = range(int(eil // n_thresh), int(eil + eil // n_thresh), int(eil // n_thresh))

DRunBiocontrolPupa = DynamicRun(
    'dd biocntrl pupa', threshold_range, trigger=pupa, action=AgregarPob(Pupal_paras['adulto'], dynamic_paras_dose)
)
DRunBiocontrolLarva = DynamicRun(
    'dd biocntrl larva', threshold_range, trigger=larvae, action=AgregarPob(Larval_paras['adulto'], dynamic_paras_dose)
)
DRunBiocontrolLarva_5 = DynamicRun(
    'dd biocntrl larva 5', threshold_range, trigger=get_larvae(5),
    action=AgregarPob(Larval_paras['adulto'], dynamic_paras_dose)
)
DRunBiocontrolLarva_45 = DynamicRun(
    'dd biocntrl larva 45', threshold_range, trigger=get_larvae([3, 4]),
    action=AgregarPob(Larval_paras['adulto'], dynamic_paras_dose)
)
DRunBiocontrolLarva_345 = DynamicRun(
    'dd biocntrl larva 345', threshold_range, trigger=get_larvae([3, 4, 5]),
    action=AgregarPob(Larval_paras['adulto'], dynamic_paras_dose)
)

# Optimised runs
ORunBiocontrolPupa3 = OptimisedRun(
    'od biocntrl pupa 3', n_pupa=3, n_larva=0, dose_total=parasitoid_dose
)
ORunBiocontrolLarvas3 = OptimisedRun(
    'od biocntrl larvas 3', n_pupa=0, n_larva=3, dose_total=parasitoid_dose
)
