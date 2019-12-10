import os
from pprint import pprint

from tikon.ejemplos.a_prioris import a_prioris
from tikon.estruc.simulador import Simulador
from tikon.exper.exper import Exper
from tikon.rae.orgs.insectos import MetamCompleta, Parasitoide
from tikon.rae.red_ae import RedAE
from tikon.rae.red_ae.obs import ObsPobs

"""
This code sets up the agroecological network for simulations. When run as a script (vs. simply importing from this 
file), it also calibrates the network model.
"""

# Create insects
Oarenosella = MetamCompleta('O. arenosella', njuvenil=5)
Larval_paras = Parasitoide('Parasitoide larvas', pupa=True)
Pupal_paras = Parasitoide('Parasitoide pupa')

# Trophic relationships
Larval_paras.parasita(Oarenosella, ['juvenil_3', 'juvenil_4', 'juvenil_5'], etp_emerg='juvenil_5')
Pupal_paras.parasita(Oarenosella, 'pupa', etp_emerg='pupa')

# Create food web
web = RedAE([Oarenosella, Larval_paras, Pupal_paras])

# A prioris for the new web
web.espec_aprioris(a_prioris)

# Observed data
dir_base = os.path.split(__file__)[0]
pobs = ObsPobs.de_csv(
    os.path.join(dir_base, 'Oarenosella_A.csv'),
    col_tiempo='Día',
    corresp={
        'Estado 1': Oarenosella['juvenil_1'],
        'Estado 2': Oarenosella['juvenil_2'],
        'Estado 3': Oarenosella['juvenil_3'],
        'Estado 4': Oarenosella['juvenil_4'],
        'Estado 5': Oarenosella['juvenil_5'],
        'Pupa': Oarenosella['pupa'],
        'Para_larva_abs': Larval_paras['juvenil'],
        'Para_pupa_abs': Pupal_paras['juvenil']
    },
    factor=655757.1429 / 500  # Convert from individuals per 500 leaflets to individuals per ha
)
exper_A = Exper('Site A', pobs)
simul = Simulador(web)

if __name__ == '__main__':
    # Calibrate
    simul.calibrar('Sitio A', exper=exper_A)

    # Save results
    simul.guardar_calib(f'Site A calibs')
    exper_A.guardar_calib(f'Site A calibs')

    # Validate and graph
    res2 = simul.simular(exper=exper_A, vars_interés=True)
    pprint(res2.validar())
    res2.graficar('imgs')
