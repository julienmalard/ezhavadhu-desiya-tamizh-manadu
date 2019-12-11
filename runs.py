import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import spotpy
import xarray as xr
from tikon.central.tiempo import Tiempo

from model import web, exper_A, Oarenosella, Larval_paras, Pupal_paras
from tikon.central import Modelo
from tikon.central.calibs import EspecCalibsCorrida
from tikon.móds.manejo import Manejo, Regla
from tikon.móds.manejo.conds import CondDía, SuperiorOIgual, CondCadaDía, Inferior
from tikon.móds.rae.manejo import AgregarPob, MultPob
from tikon.móds.rae.manejo import CondPoblación
from tikon.móds.rae.orgs.organismo import EtapaFantasma
from tikon.móds.rae.utils import EJE_ETAPA
from tikon.utils import asegurar_dir_existe

"""
This code specifies all simulation runs used in the article. It does not actually run any simulations itself, but
provides run specifications for the `analysis.py` file, where they are then run on demand (and so should not itself 
be run directly as a Python script).
All simulation settings are specified here.
"""

# Simulation settings
start_date = '1982-04-01'
final_day = 400
t_sim = Tiempo(start_date, final_day)
reps = {'paráms': 50, 'estoc': 5}

n_iter_opt = 500
n_keep_opt = 10

opt_day_range = (25, 150)
verbose = True
out_dir = 'out/runs'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Constants
parasitoid_dose = 600000
dynamic_paras_dose = parasitoid_dose / 3

eil = 655757.1429 * 0.5
survival = 0.05  # Survival rate from pesticides

# Load calibrations
web.cargar_calibs('out/Site A calibs/red')
exper_A.cargar_calibs('out/Site A calibs')

# Get all life stages
stages = web.etapas

# "Ghosts" are Tiko'n's representation of parasitised insects that will mature into parasitoids
unparasitised_larvae = [s for s in stages if s.org is Oarenosella and 'juvenil' in s.nombre]
ghosts_larvae = [s for s in stages if isinstance(s, EtapaFantasma) and s.etp_hués in unparasitised_larvae]
ghosts_pupa = [s for s in stages if isinstance(s, EtapaFantasma) and s.etp_hués == Oarenosella['pupa']]

# Quick reference to specific life stages
larvae = unparasitised_larvae + ghosts_larvae
adults = [s for s in stages if s.nombre == 'adulto']
pupa = [Oarenosella['pupa']] + ghosts_pupa
not_eggs = [s for s in stages if s.nombre != 'huevo']
nmbrs_sedent = ['pupa', 'huevo']
not_sedent = [
    s for s in stages
    if (s.etp_hués.nombre not in nmbrs_sedent if isinstance(s, EtapaFantasma) else s.nombre not in nmbrs_sedent)
]


def stringify(res):
    res.coords[EJE_ETAPA] = [str(x) for x in res.coords[EJE_ETAPA].values]
    return res


def destringify(res):
    res.coords[EJE_ETAPA] = [next((s for s in stages if str(s) == v), v) for v in res.coords[EJE_ETAPA].values]
    return res


def get_larvae(n):
    """
    Returns O. arenosella larval stages from in the model.

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
    stgs = [
               x for x in unparasitised_larvae if any(x.nombre.endswith('juvenil %i' % i) for i in n)
           ] + [
               x for x in ghosts_larvae if any(x.etp_hués.nombre.endswith('juvenil %i' % i) for i in n)
           ]
    return stgs


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

    def get_data(self, stg=None):
        """
        Obtains output data from the run. If the run has not yet been simulated, will also run the simulation.
        Otherwise, it will simply read from the previous simulation execution's output file on disk.

        Parameters
        ----------
        stg: str | list
            Output stage of interest (optional).
        Returns
        -------
        dict | np.ndarray
            Dictionnary of output variables, or numpy array of output variable if `var` was specified.
        """

        filename = self._get_output_filename()
        # Execute run if output file is not found on disk.
        if not os.path.isfile(filename):
            run((filename, (web, self.mgmt), self.all_vars))

        # Return results
        if stg is None:
            return xr.open_dataarray(filename)
        return xr.open_dataarray(filename).loc[{EJE_ETAPA: stg}]

    def _get_output_filename(self):
        """
        Returns the expected output filename for this run.

        Returns
        -------
        str
        """

        return f'{out_dir}/{self.name}.nc'


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

    def get_data(self, parallel=True, var='sum larvae'):
        """
        Obtain data from all runs. Any runs whose output files are not found on disk will be re-run.

        Parameters
        ----------
        parallel: bool
            Whether to run in parallel on multiple computer cores or not.
        var: str | list[str]
            Output variable of interest.

        Returns
        -------
        xr.DataArray
            Array of output variable. Dimension `multi` is along `self.range_`.
        """

        self.run(parallel=parallel)
        # noinspection PyTypeChecker
        return destringify(
            xr.concat(
                [xr.open_dataarray(self._get_output_filename(i)).expand_dims({'multi': [i]}) for i in self.range_],
                dim='multi'
            )
        ).loc[{EJE_ETAPA: var}]

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
                to_run[filename] = (web, self._get_mgmt(i))

        # Run all missing runs, either in parallel or else sequentially.
        if parallel:
            with Pool() as p:
                p.map(run, to_run.items())
        else:
            list(map(run, to_run.items()))

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

        return f'{out_dir}/{self.name}/{i}.nc'

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
        return Manejo(Regla(CondDía(i), self.action))


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
            _SpotPyMod(lambda x: run((None, [web, self._get_mgmt(x)])), self.n_pupa + self.n_larva),
            dbformat='ram', parallel='mpc', save_sim=False
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
                to_run[filename] = (web, self._get_mgmt(days))

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
        return Manejo([Regla(CondDía(int(t)), a) for a, t in zip(actions, days)])


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
    def objectivefunction(simulation, evaluation, params=None):
        # Note: `evaluation` and `params` parameters not used but kept for SpotPy compatibility
        return -np.log(simulation + 1)  # We want to minimise insect-days above the eil

    @staticmethod
    def _calc_above_eil(result):
        return np.sum(np.maximum(0, result.loc[{EJE_ETAPA: 'sum larvae'}] - eil)).item() / eil


def run(*args):
    args = args[0]
    if len(args) == 2:
        filename, modules = args
        all_ = False
    else:
        filename, modules, all_ = args

    if verbose and filename:
        print(f'Running {filename}')

    simul = Modelo(modules)
    res = simul.simular(final_day, reps=reps, exper=exper_A, t=t_sim, calibs=EspecCalibsCorrida(aprioris=False))

    res_final = process_results(res[str(exper_A)]['red']['Pobs'].res, all_=all_)

    if filename is None:
        return res_final
    asegurar_dir_existe(filename)
    stringify(res_final).to_netcdf(filename)


def process_results(res, all_=False):
    # Need to de- and then re-stringify in the case of pooled parallel runs to ensure that the `etapa` axis matches
    # up with actual life stage objects in `unparasitised_larvae`.
    res = destringify(stringify(res.copy()))

    # Only sum unparasitised here because ghost stages are automatically added to regular stages at end of simulations
    final = res.loc[{EJE_ETAPA: unparasitised_larvae}].sum(dim=EJE_ETAPA).expand_dims({EJE_ETAPA: ['sum larvae']})

    if all_:
        return xr.concat([final, res], dim=EJE_ETAPA)
    return final


# Simple runs
BaseRun = SingleRun('no control', mgmt=Manejo(), all_vars=True)
NoPupalParas = SingleRun(
    'without pupal paras', mgmt=Manejo(Regla(CondCadaDía(1), MultPob(Pupal_paras['adulto'], 0)))
)
NoLarvalParas = SingleRun(
    'without larval paras', mgmt=Manejo(Regla(CondCadaDía(1), MultPob(Larval_paras['adulto'], 0)))
)
NoPupalParasto150 = SingleRun(
    'without pupal paras to 150',
    mgmt=Manejo([
        Regla(CondDía(150, Inferior), MultPob(Pupal_paras['adulto'], 0)),
        Regla(CondDía(150), AgregarPob(Pupal_paras['adulto'], 100000))
    ])
)
NoLarvalParasto150 = SingleRun(
    'without larval paras to 150',
    mgmt=Manejo([
        Regla(CondDía(150, Inferior), MultPob(Larval_paras['adulto'], 0)),
        Regla(CondDía(150), AgregarPob(Larval_paras['adulto'], 100000))
    ])
)

# Time range for fixed date actions
time_range = range(1, 61, 2)

RunPesticideAdults = DateRun('fd pstcd expt adult', time_range, action=[MultPob(s, survival) for s in adults])
RunPesticideExcptEggs = DateRun('fd pstcd expt eggs', time_range, action=[MultPob(s, survival) for s in not_eggs])
RunPesticideExcptSedent = DateRun('fd pstcd expt sedent', time_range, action=[MultPob(s, survival) for s in not_sedent])
RunPesticideGeneral = DateRun('fd pstcd general', time_range, action=[MultPob(s, survival) for s in stages])
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
