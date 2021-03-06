from tikon.ecs.aprioris import APrioriDens

# A prioris for Opisina arenosella in coconut food web
from tikon.móds.rae.orgs.ecs.utils import ECS_TRANS, ECS_MRTE, ECS_ESTOC, ECS_DEPR, ECS_REPR

a_prioris = {
    'O. arenosella': [
        dict(stage='huevo',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((5, 7), 0.80),
             ),
        dict(stage='huevo',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((3, 5), 0.80),
             ),
        dict(stage='huevo',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0.15, 0.3), 0.8)
             ),
        dict(stage='huevo',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='juvenil 1',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0.10, 0.15), 0.8)
             ),
        dict(stage='juvenil 1',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((6, 8), 0.80),
             ),
        dict(stage='juvenil 1',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((1, 3), 0.80),
             ),
        dict(stage='juvenil 1',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='juvenil 2',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.1), 0.8),
             ),
        dict(stage='juvenil 2',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((8, 9), 0.80),
             ),
        dict(stage='juvenil 2',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((2, 4), 0.80)),
        dict(stage='juvenil 2',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='juvenil 3',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.01), 0.8)),
        dict(stage='juvenil 3',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((6, 8), 0.80)),
        dict(stage='juvenil 3',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((3, 5), 0.80)),
        dict(stage='juvenil 3',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='juvenil 4',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.03), 0.8)),
        dict(stage='juvenil 4',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((6, 8), 0.80)),
        dict(stage='juvenil 4',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((1, 3), 0.80)),
        dict(stage='juvenil 4',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='juvenil 5',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.03), 0.8)),
        dict(stage='juvenil 5',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((7, 9), 0.80)),
        dict(stage='juvenil 5',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((3.5, 4.5), 0.80)),
        dict(stage='juvenil 5',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='pupa',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.005), 1)),
        dict(stage='pupa',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((3, 5), 0.80)),
        dict(stage='pupa',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((2, 3), 0.80)),
        dict(stage='pupa',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.01), 0.8)),
        dict(stage='adulto',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((13, 15), 0.80)),
        dict(stage='adulto',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((1, 3), 0.80)),
        dict(stage='adulto',
             categ=ECS_REPR, sub_categ='Prob', ec='Normal', prm='n',
             apriori=APrioriDens((40, 60), 0.80)),
        dict(stage='adulto',
             categ=ECS_REPR, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((7, 8), 0.80)),
        dict(stage='adulto',
             categ=ECS_REPR, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((3, 4), 0.80)),
        dict(stage='adulto',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.05), 1.0))
    ],

    'Parasitoide larvas': [
        # Perera, P.A.C.R. A technique for laboratory mass-breeding of Eriborus trochanteratus [Hym.: Ichneumonidae]
        # a parasite of the coconut caterpillar,Nephantis serinopa [Lep.: Xylorictidae] . Entomophaga 22, 217–221
        # (1977) doi:10.1007/BF02377846
        dict(stage='adulto',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((6, 8), 0.80)),
        dict(stage='adulto',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((4, 6), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='a',
             índs=['O. arenosella', 'O. arenosella : juvenil 3'],
             apriori=APrioriDens((.005, .01), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='a',
             índs=['O. arenosella', 'O. arenosella : juvenil 4'],
             apriori=APrioriDens((.08, 0.12), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='a',
             índs=['O. arenosella', 'O. arenosella : juvenil 5'],
             apriori=APrioriDens((1.0, 1.2), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='b',
             índs=['O. arenosella', 'O. arenosella : juvenil 3'],
             apriori=APrioriDens((500, 600), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='b',
             índs=['O. arenosella', 'O. arenosella : juvenil 4'],
             apriori=APrioriDens((0, 25), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='b',
             índs=['O. arenosella', 'O. arenosella : juvenil 5'],
             apriori=APrioriDens((250, 300), 0.80)),
        dict(stage='adulto',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.01), 1)),
        dict(stage='pupa',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((8.8, 9.2), 0.80)),
        dict(stage='pupa',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 3), 0.80)),
        dict(stage='pupa',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0.004, 0.012), 0.8)),
        dict(stage='pupa',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.01), 1)),
        dict(stage='juvenil',
             categ=ECS_TRANS, sub_categ='Mult', ec='Linear', prm='a',
             apriori=APrioriDens((.8, .9), 0.80)),
        dict(stage='juvenil',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((5.5, 6.5), 0.80)),
        dict(stage='juvenil',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((1, 3), 0.80)),
        dict(stage='juvenil',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.005), 0.8)),
        dict(stage='juvenil',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.01), 1))
    ],

    'Parasitoide pupa': [
        # https://www.researchgate.net/publication/322330636_Development_of_Brachymeria_nephantidis_Gahan_Hymenoptera_Chalcididae_on_artificial_diet_reared_Opisina_arenosella
        dict(stage='adulto',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((33, 36), 0.80)),
        dict(stage='adulto',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((3, 5), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='a',
             apriori=APrioriDens((0.14, 0.15), 0.80)),
        dict(stage='adulto',
             categ=ECS_DEPR, sub_categ='Ecuación', ec='Kovai', prm='b',
             apriori=APrioriDens((150, 300), 0.80)),
        dict(stage='adulto',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.01), 1)),
        dict(stage='juvenil',
             categ=ECS_TRANS, sub_categ='Mult', ec='Linear', prm='a',
             apriori=APrioriDens((1.3, 1.4), 0.80)),
        dict(stage='juvenil',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='mu',
             apriori=APrioriDens((8, 8.5), 0.80)),
        dict(stage='juvenil',
             categ=ECS_TRANS, sub_categ='Prob', ec='Normal', prm='sigma',
             apriori=APrioriDens((2, 3), 0.80)),
        dict(stage='juvenil',
             categ=ECS_MRTE, sub_categ='Ecuación', ec='Constante', prm='q',
             apriori=APrioriDens((0, 0.03), 0.8)),
        dict(stage='juvenil',
             categ=ECS_ESTOC, sub_categ='Dist', ec='Normal', prm='sigma',
             apriori=APrioriDens((0, 0.01), 1)),
    ]
}
