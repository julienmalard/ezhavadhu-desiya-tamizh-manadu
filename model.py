import os
from pprint import pprint

from a_prioris import a_prioris
from tikon.central import Modelo, Parcela
from tikon.central.calibs import EspecCalibsCorrida
from tikon.central.exper import Exper
from tikon.central.parc import GeomParcela
from tikon.móds.rae.orgs.insectos import MetamCompleta, Parasitoide
from tikon.móds.rae.red import RedAE
from tikon.móds.rae.red.obs import ObsPobs

"""
This code sets up the agroecological network for simulations. When run as a script (vs. simply importing from this 
file), it also calibrates the network model.
"""

# Create insects
Oarenosella = MetamCompleta('O. arenosella', njuvenil=5)
Larval_paras = Parasitoide('Parasitoide larvas', pupa=True)
Pupal_paras = Parasitoide('Parasitoide pupa')

# Trophic relationships
Larval_paras.parasita(Oarenosella, ['juvenil 3', 'juvenil 4', 'juvenil 5'], etp_emerg='juvenil 5')
Pupal_paras.parasita(Oarenosella, 'pupa', etp_emerg='pupa')

# Create food web
web = RedAE([Oarenosella, Larval_paras, Pupal_paras])

# Observed data
dir_base = os.path.split(__file__)[0]
pobs = ObsPobs.de_cuadro(
    os.path.join(dir_base, 'Oarenosella_A.csv'),
    parcela='Site A',
    tiempo='Día',
    corresp={
        'Estado 1': Oarenosella['juvenil 1'],
        'Estado 2': Oarenosella['juvenil 2'],
        'Estado 3': Oarenosella['juvenil 3'],
        'Estado 4': Oarenosella['juvenil 4'],
        'Estado 5': Oarenosella['juvenil 5'],
        'Pupa': Oarenosella['pupa'],
        'Para_larva_abs': Larval_paras['juvenil'],
        'Para_pupa_abs': Pupal_paras['juvenil']
    },
    factor=655757.1429 / 500  # Convert from individuals per 500 leaflets to individuals per ha
)
exper_A = Exper('Site A', Parcela('Site A', geom=GeomParcela((7.297, 79.865))))
exper_A.datos.agregar_obs(pobs)
simul = Modelo(web)

if __name__ == '__main__':

    # A prioris for the new web
    for ins, l_aprioris in a_prioris.items():
        for d_apr in l_aprioris:
            etp = d_apr.pop('stage')
            web[ins][etp].espec_apriori(**d_apr)

    # Calibrate
    start_date = '1982-04-01'  # Perera article is unclear about precise start month, but it seems to be in 1982.
    simul.calibrar('Sitio A', exper=exper_A, t=start_date)

    # Save results
    simul.guardar_calibs('out/Site A calibs')
    exper_A.guardar_calibs('out/Site A calibs')

    # Validate and graph
    res = simul.simular(
        'valid', exper=exper_A, reps=30, t=start_date, calibs=EspecCalibsCorrida(aprioris=False), depurar=True
    )
    pprint(res.validar().a_dic())
    res.graficar('out/imgs')
