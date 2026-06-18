"""
piezo_kriging.py — Plugin principal QGIS PiezoKriging.

Gère:
  - Enregistrement du plugin / menu / toolbar
  - Orchestration du calcul Kriging et de la validation croisée
  - Création des couches QGIS (raster + contours vectoriels + points)
"""

import os
import tempfile

import numpy as np

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon, QColor, QFont
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsFeature,
    QgsGeometry, QgsPointXY, QgsField, QgsFields,
    QgsCoordinateReferenceSystem, QgsVectorFileWriter,
    QgsWkbTypes, Qgis, QgsUnitTypes,
)
from qgis.PyQt.QtCore import QVariant

from osgeo import gdal, osr

from .piezo_dialog import PiezoKrigingDialog
from .kriging_engine import (
    run_kriging, VARIOGRAM_MODELS,
    compute_experimental_variogram, fit_variogram, ordinary_kriging,
    cross_validate_loo, nice_contour_interval, find_duplicate_coords,
)


class PiezoKrigingPlugin:
    """QGIS Plugin — PiezoKriging."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dlg = None

    # ──────────────────────── Plugin lifecycle ──────────────────────

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "icons", "icon.png")
        self.action = QAction(
            QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
            "EZ Piezo",
            self.iface.mainWindow(),
        )
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&EZ Piezo", self.action)

    def unload(self):
        self.iface.removePluginMenu("&EZ Piezo", self.action)
        self.iface.removeToolBarIcon(self.action)

    # ──────────────────────── Main entry ────────────────────────────

    def run(self):
        self.dlg = PiezoKrigingDialog(self.iface.mainWindow())
        self.dlg.btn_run.clicked.connect(self._execute_kriging)
        self.dlg.btn_crossval.clicked.connect(self._execute_crossval)
        self.dlg.show()

    # ──────────────────────── Shared: build variogram ───────────────

    def _build_variogram(self, dlg, coords, values):
        """
        Fit or apply manual variogram parameters.
        Returns (vario_func, vario_params, lag_centers, gamma_exp, n_pairs).
        """
        vp = dlg.get_variogram_params()
        model_name = vp["model"]
        n_lags = vp["n_lags"]
        lag_size = vp["lag_size"] if vp["lag_size"] > 0 else None
        direction = vp["direction"]
        tolerance = vp["tolerance"]

        # Experimental variogram (always computed for display)
        lag_centers, gamma_exp, n_pairs = compute_experimental_variogram(
            coords, values,
            n_lags=n_lags,
            lag_size=lag_size,
            direction=direction if tolerance < 90.0 else None,
            tolerance=tolerance,
        )

        if vp["manual"]:
            nugget = vp["nugget"]
            if model_name == "linear":
                slope = vp["slope"]
                vario_params = {"nugget": nugget, "sill": None, "range": None, "slope": slope}

                def vario_func(h):
                    return VARIOGRAM_MODELS["linear"](np.asarray(h, dtype=float), nugget, slope)
            else:
                sill = vp["sill"]
                range_ = vp["range"]
                vario_params = {"nugget": nugget, "sill": sill, "range": range_, "slope": None}
                m = VARIOGRAM_MODELS[model_name]

                def vario_func(h):
                    return m(np.asarray(h, dtype=float), nugget, sill, range_)
        else:
            vario_params, vario_func = fit_variogram(lag_centers, gamma_exp, model_name, n_pairs)

        return vario_func, vario_params, lag_centers, gamma_exp, n_pairs, model_name

    # ──────────────────────── Kriging pipeline ─────────────────────

    def _execute_kriging(self):
        dlg = self.dlg
        names, coords, values = dlg.get_data()

        if len(values) < 3:
            QMessageBox.warning(dlg, "Données insuffisantes",
                                "Il faut au minimum 3 points pour effectuer un kriging.")
            return

        dupes = find_duplicate_coords(coords)
        if dupes:
            lines = "\n".join(f"  • {names[i]}  ↔  {names[j]}" for i, j in dupes[:10])
            if len(dupes) > 10:
                lines += f"\n  … et {len(dupes) - 10} autre(s)"
            QMessageBox.warning(
                dlg, "Coordonnées dupliquées",
                f"{len(dupes)} paire(s) de points coïncidents détectée(s) :\n\n{lines}\n\n"
                "Ces doublons rendent la matrice de krigeage singulière.\n"
                "Supprimez les doublons dans l'onglet Données avant de continuer."
            )
            return

        epsg = int(dlg.epsg_edit.text().strip() or "2154")
        _crs_check = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")
        if _crs_check.isValid() and _crs_check.mapUnits() == QgsUnitTypes.DistanceDegrees:
            QMessageBox.critical(
                dlg, "Système géographique non supporté",
                f"EPSG:{epsg} exprime les coordonnées en degrés décimaux.\n\n"
                "Le krigeage calcule des distances euclidiennes : un degré de longitude\n"
                "ne représente pas la même distance réelle qu'un degré de latitude,\n"
                "ce qui biaise le variogramme et les isopièzes résultants.\n\n"
                "Utilisez un système projeté (ex. EPSG:2154 Lambert 93, EPSG:32631 UTM 31N)."
            )
            return

        dlg.progress.setValue(5)

        try:
            vario_func, vario_params, lag_centers, gamma_exp, n_pairs, model_name = \
                self._build_variogram(dlg, coords, values)
        except Exception as e:
            QMessageBox.critical(dlg, "Erreur variogramme", str(e))
            dlg.progress.setValue(0)
            return

        dlg.progress.setValue(20)

        # Plot variogram immediately
        dlg.plot_variogram(lag_centers, gamma_exp, vario_func, vario_params,
                            n_pairs=n_pairs, model_name=model_name)
        dlg.tabs.setCurrentIndex(2)
        dlg.progress.setValue(30)

        nx, ny, pad_pct, nodata_hull = dlg.get_grid_params()
        search_params = dlg.get_search_params()
        auto_interval, manual_interval, add_labels = dlg.get_contour_params()

        try:
            # Build grid
            xmin, ymin = coords.min(axis=0)
            xmax, ymax = coords.max(axis=0)
            dx = (xmax - xmin) * pad_pct / 100.0
            dy = (ymax - ymin) * pad_pct / 100.0
            grid_x = np.linspace(xmin - dx, xmax + dx, nx)
            grid_y = np.linspace(ymin - dy, ymax + dy, ny)

            dlg.progress.setValue(40)

            Z, variance = ordinary_kriging(
                coords, values, grid_x, grid_y, vario_func,
                search_params=search_params,
            )

            dlg.progress.setValue(60)

            # NoData outside convex hull
            if nodata_hull and len(coords) >= 3:
                from scipy.spatial import Delaunay
                hull = Delaunay(coords)
                gx_m, gy_m = np.meshgrid(grid_x, grid_y)
                outside = hull.find_simplex(
                    np.column_stack([gx_m.ravel(), gy_m.ravel()])
                ) < 0
                mask = outside.reshape(ny, nx)
                Z[mask] = np.nan
                variance[mask] = np.nan

            # Auto contour interval
            z_valid = Z[np.isfinite(Z)]
            if len(z_valid) == 0:
                QMessageBox.warning(dlg, "Kriging vide",
                                    "Aucun noeud de grille ne contient de données. "
                                    "Vérifiez les paramètres de voisinage.")
                dlg.progress.setValue(0)
                return

            if auto_interval:
                contour_interval = nice_contour_interval(z_valid.max() - z_valid.min())
                dlg.contour_interval_spin.setValue(contour_interval)
            else:
                contour_interval = manual_interval

            dlg.progress.setValue(65)

            # ── Output layers ──
            tmp_dir = tempfile.mkdtemp(prefix="piezo_kriging_")
            crs = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")

            result = {"grid_x": grid_x, "grid_y": grid_y, "Z": Z}

            raster_path = os.path.join(tmp_dir, "piezo_kriging.tif")
            self._write_raster(raster_path, result, epsg)

            sigma_path = os.path.join(tmp_dir, "piezo_kriging_sigma.tif")
            sigma_result = {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "Z": np.sqrt(np.maximum(variance, 0.0)),
            }
            self._write_raster(sigma_path, sigma_result, epsg)
            dlg.progress.setValue(75)

            rlayer = QgsRasterLayer(raster_path, "Piézométrie — Kriging")
            if rlayer.isValid():
                rlayer.setCrs(crs)
                QgsProject.instance().addMapLayer(rlayer)
                self._style_raster(rlayer)

            sigma_layer = QgsRasterLayer(sigma_path, "Incertitude — Kriging (σ)")
            if sigma_layer.isValid():
                sigma_layer.setCrs(crs)
                QgsProject.instance().addMapLayer(sigma_layer)
                self._style_sigma_raster(sigma_layer)

            dlg.progress.setValue(83)

            contour_path = os.path.join(tmp_dir, "isopiezes.gpkg")
            self._generate_contours(raster_path, contour_path, contour_interval)

            vlayer_contour = QgsVectorLayer(contour_path, "Isopièzes", "ogr")
            if vlayer_contour.isValid():
                vlayer_contour.setCrs(crs)
                QgsProject.instance().addMapLayer(vlayer_contour)
                self._style_contours(vlayer_contour, add_labels)

            dlg.progress.setValue(92)
            self._create_points_layer(names, coords, values, crs)
            dlg.progress.setValue(100)

            # Success message
            if vario_params.get("slope") is not None:
                vario_str = (f"Nugget={vario_params['nugget']:.6f}  "
                             f"Pente={vario_params['slope']:.8f}")
            else:
                vario_str = (f"Nugget={vario_params['nugget']:.4f}  "
                             f"Sill={vario_params['sill']:.4f}  "
                             f"Range={vario_params['range']:.2f}")

            QMessageBox.information(
                dlg, "Succès",
                f"Kriging terminé !\n\n"
                f"• Grille : {nx}×{ny}\n"
                f"• Modèle : {model_name}\n"
                f"• {vario_str}\n"
                f"• Intervalle isopièzes : {contour_interval:.4g} m\n"
                f"• {len(values)} ouvrages interpolés\n\n"
                f"Couches créées :\n"
                f"  – Piézométrie — Kriging (Z)\n"
                f"  – Incertitude — Kriging (σ)\n"
                f"  – Isopièzes\n"
                f"  – Points ouvrages"
            )

        except Exception as e:
            QMessageBox.critical(dlg, "Erreur", f"Erreur lors du kriging :\n\n{e}")
            import traceback
            traceback.print_exc()
            dlg.progress.setValue(0)

    # ──────────────────────── Cross-validation ──────────────────────

    def _execute_crossval(self):
        dlg = self.dlg
        names, coords, values = dlg.get_data()

        if len(values) < 3:
            QMessageBox.warning(dlg, "Données insuffisantes",
                                "Il faut au minimum 3 points pour la validation croisée.")
            return

        dupes = find_duplicate_coords(coords)
        if dupes:
            lines = "\n".join(f"  • {names[i]}  ↔  {names[j]}" for i, j in dupes[:10])
            if len(dupes) > 10:
                lines += f"\n  … et {len(dupes) - 10} autre(s)"
            QMessageBox.warning(
                dlg, "Coordonnées dupliquées",
                f"{len(dupes)} paire(s) de points coïncidents détectée(s) :\n\n{lines}\n\n"
                "Ces doublons rendent la matrice de krigeage singulière.\n"
                "Supprimez les doublons dans l'onglet Données avant de continuer."
            )
            return

        try:
            vario_func, vario_params, lag_centers, gamma_exp, n_pairs, model_name = \
                self._build_variogram(dlg, coords, values)

            dlg.plot_variogram(lag_centers, gamma_exp, vario_func, vario_params,
                                n_pairs=n_pairs, model_name=model_name)

            search_params = dlg.get_search_params()
            cv = cross_validate_loo(coords, values, vario_func, search_params)
            dlg.show_crossval_results(cv, names)

            # Update variogram tab label with LOO stats
            dlg.plot_variogram(lag_centers, gamma_exp, vario_func, vario_params,
                                n_pairs=n_pairs, model_name=model_name)

        except Exception as e:
            QMessageBox.critical(dlg, "Erreur", f"Erreur lors de la validation croisée :\n\n{e}")
            import traceback
            traceback.print_exc()

    # ──────────────────────── Raster output ─────────────────────────

    @staticmethod
    def _write_raster(path, result, epsg):
        Z = result["Z"]
        grid_x = result["grid_x"]
        grid_y = result["grid_y"]
        ny, nx = Z.shape

        x_res = (grid_x[-1] - grid_x[0]) / max(nx - 1, 1)
        y_res = (grid_y[-1] - grid_y[0]) / max(ny - 1, 1)

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(path, nx, ny, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([
            grid_x[0] - x_res / 2,
            x_res,
            0,
            grid_y[-1] + y_res / 2,
            0,
            -y_res,
        ])

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())

        band = ds.GetRasterBand(1)
        Z_out = np.where(np.isfinite(Z), Z, -9999.0)
        band.WriteArray(np.flipud(Z_out).astype(np.float32))
        band.SetNoDataValue(-9999.0)
        band.FlushCache()
        ds = None

    # ──────────────────────── Contours ──────────────────────────────

    @staticmethod
    def _generate_contours(raster_path, output_path, interval):
        from osgeo import ogr
        ds = gdal.Open(raster_path)
        band = ds.GetRasterBand(1)

        gpkg_drv = ogr.GetDriverByName("GPKG")
        out_ds = gpkg_drv.CreateDataSource(output_path)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        layer = out_ds.CreateLayer("isopiezes", srs, ogr.wkbLineString)
        layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn("ELEV", ogr.OFTReal))

        gdal.ContourGenerate(band, interval, 0, [], 1, -9999.0, layer, 0, 1)

        out_ds = None
        ds = None

    # ──────────────────────── Points layer ─────────────────────────

    @staticmethod
    def _create_points_layer(names, coords, values, crs):
        vlayer = QgsVectorLayer("Point?crs=" + crs.authid(), "Ouvrages piézo", "memory")
        pr = vlayer.dataProvider()
        pr.addAttributes([
            QgsField("ouvrage", QVariant.String),
            QgsField("x", QVariant.Double),
            QgsField("y", QVariant.Double),
            QgsField("z_ngf", QVariant.Double),
        ])
        vlayer.updateFields()

        features = []
        for name, (x, y), z in zip(names, coords, values):
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))))
            f.setAttributes([name, float(x), float(y), float(z)])
            features.append(f)
        pr.addFeatures(features)
        vlayer.updateExtents()
        QgsProject.instance().addMapLayer(vlayer)

    # ──────────────────────── Styling ──────────────────────────────

    @staticmethod
    def _style_raster(layer):
        from qgis.core import (
            QgsRasterShader, QgsColorRampShader,
            QgsSingleBandPseudoColorRenderer,
        )
        provider = layer.dataProvider()
        stats = provider.bandStatistics(1)
        min_val = stats.minimumValue
        max_val = stats.maximumValue

        # Determine label decimal places based on value range
        z_range = max_val - min_val
        if z_range > 0:
            decimals = max(1, int(np.ceil(-np.log10(z_range))) + 1)
        else:
            decimals = 2
        fmt = lambda v: f"{v:.{decimals}f}"

        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)

        mid = (min_val + max_val) / 2.0
        q1 = min_val + z_range * 0.25
        q3 = min_val + z_range * 0.75

        ramp_items = [
            QgsColorRampShader.ColorRampItem(min_val, QColor(49, 54, 149), fmt(min_val)),
            QgsColorRampShader.ColorRampItem(q1, QColor(69, 117, 180), fmt(q1)),
            QgsColorRampShader.ColorRampItem(mid, QColor(255, 255, 191), fmt(mid)),
            QgsColorRampShader.ColorRampItem(q3, QColor(215, 48, 39), fmt(q3)),
            QgsColorRampShader.ColorRampItem(max_val, QColor(165, 0, 38), fmt(max_val)),
        ]
        color_ramp.setColorRampItemList(ramp_items)
        shader.setRasterShaderFunction(color_ramp)

        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    @staticmethod
    def _style_sigma_raster(layer):
        from qgis.core import (
            QgsRasterShader, QgsColorRampShader,
            QgsSingleBandPseudoColorRenderer,
        )
        provider = layer.dataProvider()
        stats = provider.bandStatistics(1)
        min_val = stats.minimumValue
        max_val = stats.maximumValue
        z_range = max_val - min_val

        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)

        mid = (min_val + max_val) / 2.0
        decimals = max(2, int(np.ceil(-np.log10(z_range))) + 1) if z_range > 0 else 3
        fmt = lambda v: f"{v:.{decimals}f}"

        ramp_items = [
            QgsColorRampShader.ColorRampItem(min_val, QColor(255, 255, 204), fmt(min_val)),
            QgsColorRampShader.ColorRampItem(mid,     QColor(253, 141,  60), fmt(mid)),
            QgsColorRampShader.ColorRampItem(max_val, QColor(128,   0,  38), fmt(max_val)),
        ]
        color_ramp.setColorRampItemList(ramp_items)
        shader.setRasterShaderFunction(color_ramp)

        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    @staticmethod
    def _style_contours(layer, add_labels=True):
        from qgis.core import (
            QgsPalLayerSettings, QgsVectorLayerSimpleLabeling,
            QgsTextFormat, QgsTextBufferSettings,
        )
        symbol = layer.renderer().symbol()
        symbol.setColor(QColor(30, 80, 160))
        symbol.setWidth(0.4)
        layer.triggerRepaint()

        if add_labels:
            settings = QgsPalLayerSettings()
            settings.fieldName = "ELEV"
            settings.placement = QgsPalLayerSettings.Line
            settings.placementFlags = QgsPalLayerSettings.OnLine

            fmt = QgsTextFormat()
            fmt.setFont(QFont("Arial", 8))
            fmt.setColor(QColor(30, 60, 140))
            buf = QgsTextBufferSettings()
            buf.setEnabled(True)
            buf.setSize(1.0)
            buf.setColor(QColor(255, 255, 255, 200))
            fmt.setBuffer(buf)
            settings.setFormat(fmt)

            labeling = QgsVectorLayerSimpleLabeling(settings)
            layer.setLabeling(labeling)
            layer.setLabelsEnabled(True)
            layer.triggerRepaint()
