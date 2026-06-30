"""
piezo_kriging.py - Main QGIS plugin entry point for EZ Piezo.

Handles:
  - Plugin registration, menu and toolbar
  - Kriging and cross-validation orchestration
  - QGIS layer creation (raster, vector contours, points)
"""

import os
import shutil
import tempfile

import numpy as np

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon, QColor, QFont
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsFeature,
    QgsGeometry, QgsPointXY, QgsField,
    QgsCoordinateReferenceSystem, Qgis, QgsUnitTypes,
    QgsArrowSymbolLayer, QgsLineSymbol, QgsSingleSymbolRenderer,
    QgsSimpleMarkerSymbolLayer, QgsMarkerSymbol,
    QgsPalLayerSettings, QgsTextFormat, QgsTextBackgroundSettings, QgsVectorLayerSimpleLabeling,
    QgsTask, QgsApplication,
)
from qgis.PyQt.QtCore import QVariant

from osgeo import gdal, osr

from .piezo_dialog import PiezoKrigingDialog
from .kriging_engine import (
    VARIOGRAM_MODELS,
    compute_experimental_variogram, fit_variogram, ordinary_kriging,
    cross_validate_loo, nice_contour_interval, find_duplicate_coords,
    compute_flow_vectors,
)


class _KrigingTask(QgsTask):
    """
    Background task: ordinary_kriging + file writing + LOO cross-validation.
    run()     → worker thread  (no Qt UI calls allowed)
    finished() → main thread   (safe to touch QGIS layers and dialog)
    """

    def __init__(self, plugin, dlg, names, coords, values,
                 vario_func, vario_params, lag_centers, gamma_exp, n_pairs, model_name,
                 grid_x, grid_y, nodata_hull, search_params,
                 epsg, crs, auto_interval, manual_interval,
                 add_labels, major_nth, major_offset):
        super().__init__("EZ Piezo — Kriging", QgsTask.CanCancel)
        self._plugin = plugin
        self._dlg = dlg
        self.names = names
        self.coords = coords
        self.values = values
        self.vario_func = vario_func
        self.vario_params = vario_params
        self.lag_centers = lag_centers
        self.gamma_exp = gamma_exp
        self.n_pairs = n_pairs
        self.model_name = model_name
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nodata_hull = nodata_hull
        self.search_params = search_params
        self.epsg = epsg
        self.crs = crs
        self.auto_interval = auto_interval
        self.manual_interval = manual_interval
        self.add_labels = add_labels
        self.major_nth = major_nth
        self.major_offset = major_offset
        # Set in run()
        self.Z = None
        self.variance = None
        self.contour_interval = manual_interval
        self.raster_path = None
        self.contour_path = None
        self.cv_result = None
        self.error_msg = None

    def run(self):
        try:
            ny, nx = len(self.grid_y), len(self.grid_x)
            self.setProgress(5)

            Z, variance = ordinary_kriging(
                self.coords, self.values, self.grid_x, self.grid_y,
                self.vario_func, search_params=self.search_params,
            )
            if self.isCanceled():
                return False
            self.setProgress(50)

            if self.nodata_hull and len(self.coords) >= 3:
                from scipy.spatial import Delaunay
                hull = Delaunay(self.coords)
                gx_m, gy_m = np.meshgrid(self.grid_x, self.grid_y)
                outside = hull.find_simplex(
                    np.column_stack([gx_m.ravel(), gy_m.ravel()])
                ) < 0
                mask = outside.reshape(ny, nx)
                Z[mask] = np.nan
                variance[mask] = np.nan

            z_valid = Z[np.isfinite(Z)]
            if len(z_valid) == 0:
                self.error_msg = "__empty_grid__"
                return False

            self.Z = Z
            self.variance = variance
            if self.auto_interval:
                self.contour_interval = nice_contour_interval(z_valid.max() - z_valid.min())
            self.setProgress(60)

            tmp_dir = tempfile.mkdtemp(prefix="piezo_kriging_")
            self._plugin._tmp_dirs.append(tmp_dir)

            self.raster_path = os.path.join(tmp_dir, "piezo_kriging.tif")
            PiezoKrigingPlugin._write_raster(
                self.raster_path,
                {"grid_x": self.grid_x, "grid_y": self.grid_y, "Z": Z},
                self.epsg,
            )
            self.setProgress(70)

            self.contour_path = os.path.join(tmp_dir, "isopiezes.gpkg")
            PiezoKrigingPlugin._generate_contours(
                self.raster_path, self.contour_path, self.contour_interval
            )
            self.setProgress(80)

            if self.isCanceled():
                return False

            self.cv_result = cross_validate_loo(
                self.coords, self.values, self.vario_func, self.search_params
            )
            self.setProgress(100)
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg = str(e)
            return False

    def finished(self, result):
        plugin = self._plugin
        dlg = self._dlg
        dlg.btn_run.setEnabled(True)
        dlg.btn_crossval.setEnabled(True)
        plugin._active_task = None

        if not result:
            dlg.progress.setValue(0)
            if self.error_msg == "__empty_grid__":
                QMessageBox.warning(
                    dlg, "Kriging vide",
                    "Aucun noeud de grille ne contient de données.\n"
                    "Vérifiez les paramètres de voisinage."
                )
            elif self.error_msg:
                QMessageBox.critical(dlg, "Erreur",
                                     f"Erreur lors du kriging :\n\n{self.error_msg}")
            return

        if self.auto_interval:
            dlg.contour_interval_spin.setValue(self.contour_interval)

        plugin._remove_layers_by_name("Kriging")
        rlayer = QgsRasterLayer(self.raster_path, "Kriging")
        if rlayer.isValid():
            rlayer.setCrs(self.crs)
            QgsProject.instance().addMapLayer(rlayer)
            plugin._style_raster(rlayer)

        plugin._flow_cache = {
            "Z": self.Z, "grid_x": self.grid_x, "grid_y": self.grid_y, "crs": self.crs
        }
        dlg.btn_refresh_flow.setEnabled(True)
        if dlg.get_flow_params()["enabled"]:
            plugin._refresh_flow_vectors()

        plugin._remove_layers_by_name("Isopièzes")
        vlayer_contour = QgsVectorLayer(self.contour_path, "Isopièzes", "ogr")
        if vlayer_contour.isValid():
            vlayer_contour.setCrs(self.crs)
            QgsProject.instance().addMapLayer(vlayer_contour)
            plugin._style_contours(vlayer_contour, self.add_labels, self.contour_interval,
                                   self.major_nth, self.major_offset)

        plugin._remove_layers_by_name("Ouvrages")
        plugin._create_points_layer(self.names, self.coords, self.values, self.crs)

        if self.cv_result is not None:
            try:
                dlg.show_crossval_results(self.cv_result, self.names)
                dlg.plot_variogram(self.lag_centers, self.gamma_exp, self.vario_func,
                                   self.vario_params, n_pairs=self.n_pairs,
                                   model_name=self.model_name)
            except Exception as _cv_err:
                import traceback as _tb
                print(f"[EZPiezo] Cross-validation display failed: {_cv_err}")
                _tb.print_exc()

        vp = self.vario_params
        if vp.get("slope") is not None:
            vario_str = f"Nugget={vp['nugget']:.6f}  Pente={vp['slope']:.8f}"
        else:
            vario_str = (f"Nugget={vp['nugget']:.4f}  "
                         f"Sill={vp['sill']:.4f}  "
                         f"Range={vp['range']:.2f}")

        flow_line = "  - Flux\n" if dlg.get_flow_params()["enabled"] else ""
        QMessageBox.information(
            dlg, "Succès",
            f"Kriging terminé !\n\n"
            f"• Grille : {len(self.grid_x)}×{len(self.grid_y)}\n"
            f"• Modèle : {self.model_name}\n"
            f"• {vario_str}\n"
            f"• Intervalle isopièzes : {self.contour_interval:.4g} m\n"
            f"• {len(self.values)} ouvrages interpolés\n\n"
            f"Couches créées :\n"
            f"  - Kriging\n"
            f"{flow_line}"
            f"  - Isopièzes\n"
            f"  - Ouvrages"
        )
        dlg.progress.setValue(100)


class _CrossvalTask(QgsTask):
    """
    Background task: LOO cross-validation only (no kriging grid).
    run()      → worker thread
    finished() → main thread
    """

    def __init__(self, plugin, dlg, names, coords, values,
                 vario_func, vario_params, lag_centers, gamma_exp, n_pairs, model_name,
                 search_params):
        super().__init__("EZ Piezo — Validation croisée", QgsTask.CanCancel)
        self._plugin = plugin
        self._dlg = dlg
        self.names = names
        self.coords = coords
        self.values = values
        self.vario_func = vario_func
        self.vario_params = vario_params
        self.lag_centers = lag_centers
        self.gamma_exp = gamma_exp
        self.n_pairs = n_pairs
        self.model_name = model_name
        self.search_params = search_params
        self.cv_result = None
        self.error_msg = None

    def run(self):
        try:
            self.setProgress(5)
            self.cv_result = cross_validate_loo(
                self.coords, self.values, self.vario_func, self.search_params
            )
            self.setProgress(100)
            return not self.isCanceled()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_msg = str(e)
            return False

    def finished(self, result):
        plugin = self._plugin
        dlg = self._dlg
        dlg.btn_run.setEnabled(True)
        dlg.btn_crossval.setEnabled(True)
        plugin._active_task = None
        dlg.progress.setValue(0)

        if not result:
            if self.error_msg:
                QMessageBox.critical(
                    dlg, "Erreur",
                    f"Erreur lors de la validation croisée :\n\n{self.error_msg}"
                )
            return

        dlg.show_crossval_results(self.cv_result, self.names)
        dlg.plot_variogram(
            self.lag_centers, self.gamma_exp, self.vario_func, self.vario_params,
            n_pairs=self.n_pairs, model_name=self.model_name,
        )


class PiezoKrigingPlugin:
    """QGIS Plugin — PiezoKriging."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dlg = None
        self._flow_cache = None
        self._tmp_dirs = []
        self._active_task = None

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
        QgsProject.instance().cleared.connect(self._cleanup_tmp)

    def unload(self):
        if self._active_task is not None:
            self._active_task.cancel()
        QgsProject.instance().cleared.disconnect(self._cleanup_tmp)
        self._cleanup_tmp()
        self.iface.removePluginMenu("&EZ Piezo", self.action)
        self.iface.removeToolBarIcon(self.action)

    def _cleanup_tmp(self):
        """Delete all temporary directories created by kriging runs."""
        for d in self._tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._tmp_dirs.clear()

    # ──────────────────────── Main entry ────────────────────────────

    def run(self):
        if self.dlg is not None and self.dlg.isVisible():
            self.dlg.raise_()
            self.dlg.activateWindow()
            return
        self.dlg = PiezoKrigingDialog(self.iface.mainWindow())
        self.dlg.btn_run.clicked.connect(self._execute_kriging)
        self.dlg.btn_crossval.clicked.connect(self._execute_crossval)
        self.dlg.btn_refresh_flow.clicked.connect(self._refresh_flow_vectors)
        self.dlg.btn_restyle_contours.clicked.connect(self._restyle_contours)
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
            min_pairs=vp["min_pairs"],
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
            vario_params, vario_func = fit_variogram(
                lag_centers, gamma_exp, model_name, n_pairs,
                force_nugget_zero=vp["force_nugget_zero"],
            )

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

        # ── EPSG validation (issues 2 & 3) ──
        epsg_str = dlg.epsg_edit.text().strip() or "2154"
        try:
            epsg = int(epsg_str)
        except ValueError:
            QMessageBox.critical(dlg, "EPSG invalide",
                                 f"'{epsg_str}' n'est pas un entier valide.")
            return

        crs = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")
        if not crs.isValid():
            QMessageBox.critical(
                dlg, "EPSG invalide",
                f"EPSG:{epsg} n'est pas reconnu par QGIS.\n"
                "Utilisez un code projeté valide (ex. EPSG:2154 Lambert 93, EPSG:32631 UTM 31N)."
            )
            return
        if crs.mapUnits() == QgsUnitTypes.DistanceDegrees:
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
        dlg.plot_variogram(lag_centers, gamma_exp, vario_func, vario_params,
                           n_pairs=n_pairs, model_name=model_name)
        dlg.tabs.setCurrentIndex(2)
        dlg.progress.setValue(30)

        nx, ny, pad_pct, nodata_hull = dlg.get_grid_params()
        search_params = dlg.get_search_params()
        auto_interval, manual_interval, add_labels, major_nth, major_offset = dlg.get_contour_params()

        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        dx = (xmax - xmin) * pad_pct / 100.0
        dy = (ymax - ymin) * pad_pct / 100.0
        grid_x = np.linspace(xmin - dx, xmax + dx, nx)
        grid_y = np.linspace(ymin - dy, ymax + dy, ny)

        # ── Launch background task ──
        dlg.btn_run.setEnabled(False)
        dlg.btn_crossval.setEnabled(False)
        task = _KrigingTask(
            plugin=self, dlg=dlg,
            names=names, coords=coords, values=values,
            vario_func=vario_func, vario_params=vario_params,
            lag_centers=lag_centers, gamma_exp=gamma_exp,
            n_pairs=n_pairs, model_name=model_name,
            grid_x=grid_x, grid_y=grid_y,
            nodata_hull=nodata_hull, search_params=search_params,
            epsg=epsg, crs=crs,
            auto_interval=auto_interval, manual_interval=manual_interval,
            add_labels=add_labels, major_nth=major_nth, major_offset=major_offset,
        )
        task.progressChanged.connect(lambda p: dlg.progress.setValue(int(p)))
        self._active_task = task
        QgsApplication.taskManager().addTask(task)

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
        except Exception as e:
            QMessageBox.critical(dlg, "Erreur variogramme", str(e))
            return

        search_params = dlg.get_search_params()

        dlg.btn_run.setEnabled(False)
        dlg.btn_crossval.setEnabled(False)
        task = _CrossvalTask(
            plugin=self, dlg=dlg,
            names=names, coords=coords, values=values,
            vario_func=vario_func, vario_params=vario_params,
            lag_centers=lag_centers, gamma_exp=gamma_exp,
            n_pairs=n_pairs, model_name=model_name,
            search_params=search_params,
        )
        task.progressChanged.connect(lambda p: dlg.progress.setValue(int(p)))
        self._active_task = task
        QgsApplication.taskManager().addTask(task)

    # ──────────────────────── Layer helpers ─────────────────────────

    @staticmethod
    def _remove_layers_by_name(name):
        """Remove all map layers whose name matches exactly."""
        for lyr in QgsProject.instance().mapLayersByName(name):
            QgsProject.instance().removeMapLayer(lyr.id())

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
        vlayer = QgsVectorLayer("Point?crs=" + crs.authid(), "Ouvrages", "memory")
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
        PiezoKrigingPlugin._style_points_layer(vlayer)

    @staticmethod
    def _style_points_layer(layer):
        marker_sl = QgsSimpleMarkerSymbolLayer()
        marker_sl.setShape(QgsSimpleMarkerSymbolLayer.Cross)
        marker_sl.setSize(5.0)
        marker_sl.setColor(QColor(0, 0, 0))
        marker_sl.setStrokeColor(QColor(0, 0, 0))
        marker_sl.setStrokeWidth(0.7)
        symbol = QgsMarkerSymbol([marker_sl])
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        settings = QgsPalLayerSettings()
        settings.fieldName = "ouvrage"
        settings.enabled = True
        # Qgis.LabelPlacement introduced in QGIS 3.26; fallback for 3.16
        if hasattr(Qgis, "LabelPlacement"):
            settings.placement = Qgis.LabelPlacement.OverPoint
        else:
            settings.placement = 0  # OverPoint
        settings.quadOffset = QgsPalLayerSettings.QuadrantAboveRight
        settings.xOffset = 3.0
        settings.yOffset = -2.0

        bg = QgsTextBackgroundSettings()
        bg.setEnabled(True)
        bg.setType(QgsTextBackgroundSettings.ShapeRectangle)
        bg.setFillColor(QColor(255, 255, 255))
        bg.setStrokeColor(QColor(255, 255, 255))
        bg.setOpacity(1.0)

        tf = QgsTextFormat()
        tf.setFont(QFont("Arial", 8))
        tf.setSize(8)
        tf.setBackground(bg)
        settings.setFormat(tf)

        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)
        layer.triggerRepaint()

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

        def fmt(v):
            return f"{v:.{decimals}f}"

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

        def fmt(v):
            return f"{v:.{decimals}f}"

        ramp_items = [
            QgsColorRampShader.ColorRampItem(min_val, QColor(255, 255, 204), fmt(min_val)),
            QgsColorRampShader.ColorRampItem(mid, QColor(253, 141, 60), fmt(mid)),
            QgsColorRampShader.ColorRampItem(max_val, QColor(128, 0, 38), fmt(max_val)),
        ]
        color_ramp.setColorRampItemList(ramp_items)
        shader.setRasterShaderFunction(color_ramp)

        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    def _refresh_flow_vectors(self):
        """Regenerate the flow vectors layer from cached grid data (no re-kriging)."""
        if self._flow_cache is None or self.dlg is None:
            return
        flow_params = self.dlg.get_flow_params()
        if not flow_params["enabled"]:
            return

        cache = self._flow_cache
        vectors = compute_flow_vectors(
            cache["Z"], cache["grid_x"], cache["grid_y"],
            step_x=flow_params["step_x"],
            step_y=flow_params["step_y"],
        )

        for lyr in QgsProject.instance().mapLayersByName("Flux"):
            QgsProject.instance().removeMapLayer(lyr.id())

        if vectors:
            flow_layer = self._create_flow_vectors_layer(
                vectors, cache["crs"],
                flow_params["step_x"], flow_params["step_y"],
                cache["grid_x"], cache["grid_y"],
            )
            if flow_layer.isValid():
                flow_layer.setCrs(cache["crs"])
                QgsProject.instance().addMapLayer(flow_layer)
                self._style_flow_vectors(flow_layer)

    @staticmethod
    def _create_flow_vectors_layer(vectors, crs, step_x, step_y, grid_x, grid_y):
        """Build an in-memory LineString layer of normalised flow arrows."""
        vlayer = QgsVectorLayer(
            f"LineString?crs={crs.authid()}", "Flux", "memory"
        )
        pr = vlayer.dataProvider()
        pr.addAttributes([
            QgsField("magnitude", QVariant.Double),
            QgsField("angle_deg", QVariant.Double),
        ])
        vlayer.updateFields()

        nx = len(grid_x)
        ny = len(grid_y)
        dx_sp = (grid_x[-1] - grid_x[0]) / max(nx - 1, 1)
        dy_sp = (grid_y[-1] - grid_y[0]) / max(ny - 1, 1)
        scale = min(step_x * dx_sp, step_y * dy_sp) * 0.2

        features = []
        for v in vectors:
            x0, y0 = v["x"], v["y"]
            x1 = x0 + v["dx"] * scale
            y1 = y0 + v["dy"] * scale
            angle = float(np.degrees(np.arctan2(v["dy"], v["dx"])))
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x0, y0), QgsPointXY(x1, y1)]))
            f.setAttributes([v["magnitude"], angle])
            features.append(f)

        pr.addFeatures(features)
        vlayer.updateExtents()
        return vlayer

    @staticmethod
    def _style_flow_vectors(layer):
        """Style flow vector layer as filled rectangle-shaft + triangle-head arrows."""
        arrow_sl = QgsArrowSymbolLayer()
        arrow_sl.setIsCurved(False)
        arrow_sl.setIsRepeated(False)
        arrow_sl.setArrowWidth(1.4)      # mm — shaft width (rectangle)
        arrow_sl.setHeadLength(2.8)      # mm — triangle length
        arrow_sl.setHeadThickness(2.2)   # mm — triangle base width
        arrow_sl.setColor(QColor(0, 0, 0))
        arrow_sl.setFillColor(QColor(0, 0, 0))
        symbol = QgsLineSymbol([arrow_sl])
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()

    def _restyle_contours(self):
        layers = QgsProject.instance().mapLayersByName("Isopièzes")
        if not layers:
            QMessageBox.warning(self.dlg, "Couche introuvable",
                                "Aucune couche 'Isopièzes' trouvée dans le projet.")
            return
        _, contour_interval, add_labels, major_nth, major_offset = self.dlg.get_contour_params()
        for lyr in layers:
            self._style_contours(lyr, add_labels, contour_interval, major_nth, major_offset)
            lyr.triggerRepaint()

    @staticmethod
    def _style_contours(layer, add_labels=True, contour_interval=1.0, major_nth=5, major_offset=0.0):
        from qgis.core import (
            QgsPalLayerSettings, QgsTextFormat, QgsTextBufferSettings,
            QgsRuleBasedRenderer, QgsRuleBasedLabeling,
            QgsLineSymbol,
        )

        major_interval = contour_interval * major_nth
        eps = major_interval * 0.001
        shifted = f'("ELEV" - {major_offset:.8g})'
        major_expr = (
            f'abs({shifted} - round({shifted} / {major_interval:.8g}) * {major_interval:.8g}) < {eps:.8g}'
        )

        # Rule-based renderer: major (thick) vs minor (thin)
        major_sym = QgsLineSymbol.createSimple({
            'line_color': '30,80,160,255',
            'line_width': '0.65',
        })
        minor_sym = QgsLineSymbol.createSimple({
            'line_color': '30,80,160,160',
            'line_width': '0.22',
        })

        major_rule = QgsRuleBasedRenderer.Rule(major_sym)
        major_rule.setFilterExpression(major_expr)
        major_rule.setLabel("Isopièze principale")

        minor_rule = QgsRuleBasedRenderer.Rule(minor_sym)
        minor_rule.setFilterExpression(f'NOT ({major_expr})')
        minor_rule.setLabel("Isopièze secondaire")

        root_rule = QgsRuleBasedRenderer.Rule(None)
        root_rule.appendChild(major_rule)
        root_rule.appendChild(minor_rule)

        layer.setRenderer(QgsRuleBasedRenderer(root_rule))

        # Rule-based labeling: only major contours get elevation labels
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

            label_rule = QgsRuleBasedLabeling.Rule(settings)
            label_rule.setFilterExpression(major_expr)
            label_rule.setActive(True)

            root_label = QgsRuleBasedLabeling.Rule(None)
            root_label.appendChild(label_rule)

            layer.setLabeling(QgsRuleBasedLabeling(root_label))
            layer.setLabelsEnabled(True)

        layer.triggerRepaint()
