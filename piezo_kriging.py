"""
piezo_kriging.py - Main QGIS plugin entry point for EZ Piezo.

Handles:
  - Plugin registration, menu and toolbar
  - Kriging and cross-validation orchestration
  - QGIS layer creation (raster, vector contours, points)
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
    QgsArrowSymbolLayer, QgsLineSymbol, QgsSingleSymbolRenderer,
    QgsSimpleMarkerSymbolLayer, QgsMarkerSymbol,
    QgsPalLayerSettings, QgsTextFormat, QgsTextBackgroundSettings, QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QVariant

from osgeo import gdal, osr

from .piezo_dialog import PiezoKrigingDialog
from .kriging_engine import (
    run_kriging, VARIOGRAM_MODELS,
    compute_experimental_variogram, fit_variogram, ordinary_kriging,
    cross_validate_loo, nice_contour_interval, find_duplicate_coords,
    compute_flow_vectors,
)


class PiezoKrigingPlugin:
    """QGIS Plugin — PiezoKriging."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dlg = None
        self._flow_cache = None  # {Z, grid_x, grid_y, crs} — updated after each kriging run

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
        auto_interval, manual_interval, add_labels, major_nth, major_offset = dlg.get_contour_params()

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

            dlg.progress.setValue(75)

            rlayer = QgsRasterLayer(raster_path, "Kriging")
            if rlayer.isValid():
                rlayer.setCrs(crs)
                QgsProject.instance().addMapLayer(rlayer)
                self._style_raster(rlayer)

            # Flow vectors - cache grid for dynamic refresh without re-kriging
            self._flow_cache = {
                "Z": Z, "grid_x": grid_x, "grid_y": grid_y, "crs": crs
            }
            dlg.btn_refresh_flow.setEnabled(True)
            if dlg.get_flow_params()["enabled"]:
                self._refresh_flow_vectors()

            dlg.progress.setValue(83)

            contour_path = os.path.join(tmp_dir, "isopiezes.gpkg")
            self._generate_contours(raster_path, contour_path, contour_interval)

            vlayer_contour = QgsVectorLayer(contour_path, "Isopièzes", "ogr")
            if vlayer_contour.isValid():
                vlayer_contour.setCrs(crs)
                QgsProject.instance().addMapLayer(vlayer_contour)
                self._style_contours(vlayer_contour, add_labels, contour_interval, major_nth, major_offset)

            dlg.progress.setValue(92)
            self._create_points_layer(names, coords, values, crs)
            dlg.progress.setValue(100)

            # Auto cross-validation
            try:
                cv = cross_validate_loo(coords, values, vario_func, search_params)
                dlg.show_crossval_results(cv, names)
                dlg.plot_variogram(lag_centers, gamma_exp, vario_func, vario_params,
                                   n_pairs=n_pairs, model_name=model_name)
            except Exception as _cv_err:
                import traceback as _tb
                print(f"[EZPiezo] Automatic cross-validation failed: {_cv_err}")
                _tb.print_exc()

            # Success message
            if vario_params.get("slope") is not None:
                vario_str = (f"Nugget={vario_params['nugget']:.6f}  "
                             f"Pente={vario_params['slope']:.8f}")
            else:
                vario_str = (f"Nugget={vario_params['nugget']:.4f}  "
                             f"Sill={vario_params['sill']:.4f}  "
                             f"Range={vario_params['range']:.2f}")

            flow_line = "  - Flux\n" if dlg.get_flow_params()["enabled"] else ""
            QMessageBox.information(
                dlg, "Succès",
                f"Kriging terminé !\n\n"
                f"• Grille : {nx}×{ny}\n"
                f"• Modèle : {model_name}\n"
                f"• {vario_str}\n"
                f"• Intervalle isopièzes : {contour_interval:.4g} m\n"
                f"• {len(values)} ouvrages interpolés\n\n"
                f"Couches créées :\n"
                f"  - Kriging\n"
                f"{flow_line}"
                f"  - Isopièzes\n"
                f"  - Ouvrages"
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
