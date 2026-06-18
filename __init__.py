"""
EZ Piezo – Plugin QGIS pour cartes piézométriques par Kriging
"""


def classFactory(iface):
    from .piezo_kriging import PiezoKrigingPlugin
    return PiezoKrigingPlugin(iface)
