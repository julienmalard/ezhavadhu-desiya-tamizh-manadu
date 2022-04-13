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
இந்த குறியீட்டை மூலம் உருவகப்படுத்துதலுக்காக உணவு வலை உருவாக்கப்படும். இந்த கோப்பை நேரடியாக இயக்கினால், உணவு வலை 
மாதிரியை அளவீடும். இந்த கோப்பிலிருந்து எற்றுமதி செய்தால், அளவீடு ஆகாது.
"""

# பூச்சி உருவாக்கம்
கருந்தலைப்புழு = MetamCompleta('O. arenosella', njuvenil=5)
புழு_ஒட்டுண்ணி = Parasitoide('Parasitoide larvas', pupa=True)
கூட்டுப்புழு_ஒட்டுண்ணி = Parasitoide('Parasitoide pupa')

# உணவு வலை உருவாக்கம்
வலை = RedAE([கருந்தலைப்புழு, புழு_ஒட்டுண்ணி, கூட்டுப்புழு_ஒட்டுண்ணி])

# உணவு வலை தொடர்புகள்
புழு_ஒட்டுண்ணி.parasita(கருந்தலைப்புழு, ['juvenil 3', 'juvenil 4', 'juvenil 5'], etp_emerg='juvenil 5')
கூட்டுப்புழு_ஒட்டுண்ணி.parasita(கருந்தலைப்புழு, 'pupa', etp_emerg='pupa')

# கண்டறியப்பட்ட தரவுகள்
மூல்_கோப்புரை = os.path.split(__file__)[0]
மக்கள்தொகை = ObsPobs.de_cuadro(
    os.path.join(மூல்_கோப்புரை, 'கருந்தலைப்புழு-அ.csv'),
    parcela='வயல் அ',
    tiempo='நாள்',
    corresp={
        'குடம்பி ௧': கருந்தலைப்புழு['juvenil 1'],
        'குடம்பி ௨': கருந்தலைப்புழு['juvenil 2'],
        'குடம்பி ௩': கருந்தலைப்புழு['juvenil 3'],
        'குடம்பி ௪': கருந்தலைப்புழு['juvenil 4'],
        'குடம்பி ௫': கருந்தலைப்புழு['juvenil 5'],
        'கூட்டுப்புழு': கருந்தலைப்புழு['pupa'],
        'எண்_குடம்பி_ஒட்டுண்ணி': புழு_ஒட்டுண்ணி['juvenil'],
        'எண்_கூட்டுப்புழு_ஒட்டுண்ணி': கூட்டுப்புழு_ஒட்டுண்ணி['juvenil']
    },
    factor=655757.1429 / 500  # எண்ணைக்கையை கூட்டி இலை மூலம் ஹெக்டேருக்கு மாற்றம்
)
வயில்_அ = Exper('Site A', Parcela('Site A', geom=GeomParcela((7.297, 79.865))))
வயில்_அ.datos.agregar_obs(மக்கள்தொகை)
மாதிரி = Modelo(வலை)

if __name__ == '__main__':

    # புது உணவு வலைக்காக ஆரம்ப நிகழ்வெண் பரவல்கள்
    for பூச்சி, நிகழ்வெண்_பட்டியல் in a_prioris.items():
        for நிகழ்வெண்_அகராதி in நிகழ்வெண்_பட்டியல்:
            படி = நிகழ்வெண்_அகராதி.pop('stage')
            வலை[பூச்சி][படி].espec_apriori(**நிகழ்வெண்_அகராதி)

    # அளவீடு
    # (Perera கட்டுரையில் ஆரம்பு தேதி தெளிவாக குறிப்பட்டப்பட்டதில்லை. ஆனால் ௲௯௱௮௰௨ ஆண்டில் தான் என்று தோன்றுகிறது.)
    ஆரம்பு_தேதி = '1982-04-01'
    மாதிரி.calibrar('வயல் அ', exper=வயில்_அ, t=ஆரம்பு_தேதி)

    # விளைவு சேமிப்பு
    மாதிரி.guardar_calibs('வெளியீடு/அளவீடு வயில் அ')
    வயில்_அ.guardar_calibs('வெளியீடு/அளவீடு வயில் அ')

    # சரிபார்த்தல் மற்றும் வரைதல்
    விளைவுகள் = மாதிரி.simular(
        'valid', exper=வயில்_அ, reps=30, t=ஆரம்பு_தேதி, calibs=EspecCalibsCorrida(aprioris=False), depurar=True
    )
    pprint(விளைவுகள்.validar().a_dic())
    விளைவுகள்.graficar('வெளியீடு/உருப்படங்கள்')
