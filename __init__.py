"""
EZ Piezo - QGIS plugin for piezometric mapping by Kriging interpolation.
"""


def classFactory(iface):
    from .piezo_kriging import PiezoKrigingPlugin
    return PiezoKrigingPlugin(iface)
