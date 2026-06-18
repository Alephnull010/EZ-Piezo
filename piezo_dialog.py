"""
piezo_dialog.py — Interface graphique Qt pour PiezoKriging.
"""

import os
import csv

import numpy as np

from qgis.PyQt.QtCore import Qt, QVariant
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QTabWidget, QWidget, QProgressBar,
    QHeaderView, QCheckBox, QSplitter, QFrame, QRadioButton,
    QButtonGroup, QSizePolicy,
)
from qgis.PyQt.QtGui import QFont, QColor, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PiezoKrigingDialog(QDialog):
    """Main dialog for PiezoKriging plugin."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PiezoKriging — Carte piézométrique par Kriging")
        self.setMinimumSize(1000, 720)
        self.resize(1100, 800)

        # State for column mapping
        self._raw_rows = []
        self._raw_headers = []
        # State for LOO stats (updated after cross-validation)
        self._loo_rmse = None
        self._loo_rmsse = None

        self._build_ui()

    # ─────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        title = QLabel("PiezoKriging")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c5f8a;")
        subtitle = QLabel("Interpolation piézométrique par Ordinary Kriging")
        subtitle.setStyleSheet("font-size: 11px; color: #666; margin-bottom: 8px;")
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._build_tab_data()
        self._build_tab_params()
        self._build_tab_vario()
        self._build_tab_crossval()

        self.progress = QProgressBar()
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Lancer le Kriging")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #2c7fb8; color: white; font-size: 14px; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a5a8a; }"
        )
        run_row.addStretch()
        run_row.addWidget(self.btn_run)
        run_row.addStretch()
        main_layout.addLayout(run_row)

    # ── Tab 1: Données ────────────────────────────────────────────────

    def _build_tab_data(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File row
        file_row = QHBoxLayout()
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("Chemin vers fichier CSV …")
        btn_browse = QPushButton("Parcourir…")
        btn_browse.clicked.connect(self._browse_csv)
        btn_load = QPushButton("Charger CSV")
        btn_load.clicked.connect(self._load_csv)
        file_row.addWidget(self.csv_path_edit, 4)
        file_row.addWidget(btn_browse, 1)
        file_row.addWidget(btn_load, 1)
        layout.addLayout(file_row)

        # Separator + header
        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("Séparateur :"))
        self.sep_combo = QComboBox()
        self.sep_combo.addItems([";", ",", "TAB", "|"])
        sep_row.addWidget(self.sep_combo)
        sep_row.addSpacing(20)
        self.header_check = QCheckBox("Première ligne = en-tête")
        self.header_check.setChecked(True)
        sep_row.addWidget(self.header_check)
        sep_row.addStretch()
        layout.addLayout(sep_row)

        # Column mapping (shown after load)
        self._mapping_frame = QFrame()
        self._mapping_frame.setFrameShape(QFrame.StyledPanel)
        map_layout = QHBoxLayout(self._mapping_frame)
        map_layout.addWidget(QLabel("Mapping :"))
        self._col_labels = ["Ouvrage", "X", "Y", "Z (m)"]
        self._col_combos = []
        for lbl in self._col_labels:
            map_layout.addWidget(QLabel(f"{lbl} →"))
            cb = QComboBox()
            cb.setMinimumWidth(120)
            cb.addItem("(auto)")
            self._col_combos.append(cb)
            map_layout.addWidget(cb)
        btn_apply = QPushButton("Appliquer")
        btn_apply.clicked.connect(self._apply_mapping)
        map_layout.addWidget(btn_apply)
        map_layout.addStretch()
        self._mapping_frame.setVisible(False)
        layout.addWidget(self._mapping_frame)

        # Data table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Ouvrage", "X", "Y", "Z (m NGF)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Add / remove buttons
        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Ajouter ligne")
        btn_add.clicked.connect(self._add_row)
        btn_del = QPushButton("− Supprimer sélection")
        btn_del.clicked.connect(self._delete_selected)
        btn_clear = QPushButton("Tout effacer")
        btn_clear.clicked.connect(self._clear_table)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "1 — Données")

    # ── Tab 2: Paramètres ─────────────────────────────────────────────

    def _build_tab_params(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Variogramme ──
        grp_vario = QGroupBox("Variogramme")
        gv = QVBoxLayout(grp_vario)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Modèle :"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["spherical", "exponential", "gaussian", "linear"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row1.addWidget(self.model_combo)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Nb lags :"))
        self.n_lags_spin = QSpinBox()
        self.n_lags_spin.setRange(5, 100)
        self.n_lags_spin.setValue(15)
        row1.addWidget(self.n_lags_spin)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Lag size (0=auto) :"))
        self.lag_size_spin = QDoubleSpinBox()
        self.lag_size_spin.setRange(0.0, 9999999.0)
        self.lag_size_spin.setDecimals(4)
        self.lag_size_spin.setValue(0.0)
        self.lag_size_spin.setSpecialValueText("auto")
        row1.addWidget(self.lag_size_spin)
        row1.addStretch()
        gv.addLayout(row1)

        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Direction (°, depuis Est) :"))
        self.direction_spin = QDoubleSpinBox()
        self.direction_spin.setRange(-180.0, 180.0)
        self.direction_spin.setValue(0.0)
        self.direction_spin.setDecimals(1)
        self.direction_spin.setToolTip("0° = Est, 90° = Nord. Ignoré si tolérance = 90°.")
        row_dir.addWidget(self.direction_spin)
        row_dir.addSpacing(12)
        row_dir.addWidget(QLabel("Tolérance (°) :"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.1, 90.0)
        self.tolerance_spin.setValue(90.0)
        self.tolerance_spin.setDecimals(1)
        self.tolerance_spin.setToolTip("90° = omnidirectionnel (comportement par défaut).")
        row_dir.addWidget(self.tolerance_spin)
        row_dir.addStretch()
        gv.addLayout(row_dir)

        # Manual params row
        row_manual = QHBoxLayout()
        self.manual_vario_check = QCheckBox("Paramètres manuels")
        self.manual_vario_check.toggled.connect(self._on_manual_toggle)
        row_manual.addWidget(self.manual_vario_check)

        row_manual.addWidget(QLabel("Nugget :"))
        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0.0, 999999.0)
        self.nugget_spin.setDecimals(6)
        self.nugget_spin.setValue(0.0)
        self.nugget_spin.setEnabled(False)
        row_manual.addWidget(self.nugget_spin)

        self._sill_label = QLabel("Sill :")
        row_manual.addWidget(self._sill_label)
        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0.001, 999999.0)
        self.sill_spin.setDecimals(6)
        self.sill_spin.setValue(1.0)
        self.sill_spin.setEnabled(False)
        row_manual.addWidget(self.sill_spin)

        self._range_label = QLabel("Range :")
        row_manual.addWidget(self._range_label)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(0.001, 9999999.0)
        self.range_spin.setDecimals(4)
        self.range_spin.setValue(1000.0)
        self.range_spin.setEnabled(False)
        row_manual.addWidget(self.range_spin)

        self._slope_label = QLabel("Pente :")
        self._slope_label.setVisible(False)
        row_manual.addWidget(self._slope_label)
        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setRange(1e-10, 999999.0)
        self.slope_spin.setDecimals(8)
        self.slope_spin.setValue(0.001)
        self.slope_spin.setEnabled(False)
        self.slope_spin.setVisible(False)
        row_manual.addWidget(self.slope_spin)

        gv.addLayout(row_manual)
        layout.addWidget(grp_vario)

        # ── Grille ──
        grp_grid = QGroupBox("Grille d'interpolation")
        gg = QVBoxLayout(grp_grid)

        row_g1 = QHBoxLayout()
        row_g1.addWidget(QLabel("Noeuds X :"))
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 2000)
        self.nx_spin.setValue(100)
        self.nx_spin.valueChanged.connect(self._update_spacing_label)
        row_g1.addWidget(self.nx_spin)
        row_g1.addSpacing(12)
        row_g1.addWidget(QLabel("Noeuds Y :"))
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 2000)
        self.ny_spin.setValue(100)
        self.ny_spin.valueChanged.connect(self._update_spacing_label)
        row_g1.addWidget(self.ny_spin)
        row_g1.addSpacing(12)
        row_g1.addWidget(QLabel("Marge (%) :"))
        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0.0, 50.0)
        self.pad_spin.setValue(5.0)
        self.pad_spin.valueChanged.connect(self._update_spacing_label)
        row_g1.addWidget(self.pad_spin)
        row_g1.addStretch()
        gg.addLayout(row_g1)

        row_g2 = QHBoxLayout()
        self._spacing_label = QLabel("Espacement X : —  |  Y : —")
        self._spacing_label.setStyleSheet("color: #555; font-style: italic;")
        row_g2.addWidget(self._spacing_label)
        row_g2.addStretch()
        self.hull_check = QCheckBox("NoData hors convex hull")
        self.hull_check.setToolTip(
            "Marque comme NoData les noeuds de grille en dehors de l'enveloppe convexe des points."
        )
        row_g2.addWidget(self.hull_check)
        gg.addLayout(row_g2)

        layout.addWidget(grp_grid)

        # ── Isopièzes ──
        grp_contour = QGroupBox("Courbes isopièzes")
        gc = QHBoxLayout(grp_contour)
        self.auto_interval_check = QCheckBox("Intervalle auto")
        self.auto_interval_check.setChecked(True)
        self.auto_interval_check.toggled.connect(self._on_auto_interval_toggle)
        gc.addWidget(self.auto_interval_check)
        gc.addWidget(QLabel("Intervalle (m) :"))
        self.contour_interval_spin = QDoubleSpinBox()
        self.contour_interval_spin.setRange(0.0001, 10000.0)
        self.contour_interval_spin.setValue(1.0)
        self.contour_interval_spin.setDecimals(4)
        self.contour_interval_spin.setEnabled(False)
        gc.addWidget(self.contour_interval_spin)
        gc.addSpacing(20)
        self.add_labels_check = QCheckBox("Étiquettes sur isopièzes")
        self.add_labels_check.setChecked(True)
        gc.addWidget(self.add_labels_check)
        gc.addStretch()
        layout.addWidget(grp_contour)

        # ── Voisinage de recherche ──
        grp_search = QGroupBox("Voisinage de recherche")
        gs = QVBoxLayout(grp_search)

        self._search_btn_group = QButtonGroup(self)
        self.search_all_radio = QRadioButton("Toutes les données")
        self.search_ell_radio = QRadioButton("Ellipse de recherche")
        self.search_all_radio.setChecked(True)
        self._search_btn_group.addButton(self.search_all_radio)
        self._search_btn_group.addButton(self.search_ell_radio)
        self.search_ell_radio.toggled.connect(self._on_search_mode_changed)

        row_sr = QHBoxLayout()
        row_sr.addWidget(self.search_all_radio)
        row_sr.addSpacing(30)
        row_sr.addWidget(self.search_ell_radio)
        row_sr.addStretch()
        gs.addLayout(row_sr)

        self._ell_frame = QFrame()
        ell_layout = QHBoxLayout(self._ell_frame)
        ell_layout.setContentsMargins(0, 0, 0, 0)

        ell_layout.addWidget(QLabel("Rayon 1 :"))
        self.r1_spin = QDoubleSpinBox()
        self.r1_spin.setRange(0.001, 9999999.0)
        self.r1_spin.setDecimals(2)
        self.r1_spin.setValue(1000.0)
        ell_layout.addWidget(self.r1_spin)

        ell_layout.addSpacing(8)
        ell_layout.addWidget(QLabel("Rayon 2 :"))
        self.r2_spin = QDoubleSpinBox()
        self.r2_spin.setRange(0.001, 9999999.0)
        self.r2_spin.setDecimals(2)
        self.r2_spin.setValue(1000.0)
        ell_layout.addWidget(self.r2_spin)

        ell_layout.addSpacing(8)
        ell_layout.addWidget(QLabel("Angle (°) :"))
        self.search_angle_spin = QDoubleSpinBox()
        self.search_angle_spin.setRange(-180.0, 180.0)
        self.search_angle_spin.setDecimals(1)
        self.search_angle_spin.setValue(0.0)
        ell_layout.addWidget(self.search_angle_spin)

        ell_layout.addSpacing(8)
        ell_layout.addWidget(QLabel("Min données :"))
        self.min_nb_spin = QSpinBox()
        self.min_nb_spin.setRange(1, 999)
        self.min_nb_spin.setValue(1)
        ell_layout.addWidget(self.min_nb_spin)

        ell_layout.addSpacing(8)
        ell_layout.addWidget(QLabel("Max données :"))
        self.max_nb_spin = QSpinBox()
        self.max_nb_spin.setRange(1, 9999)
        self.max_nb_spin.setValue(20)
        ell_layout.addWidget(self.max_nb_spin)

        ell_layout.addStretch()
        self._ell_frame.setEnabled(False)
        gs.addWidget(self._ell_frame)
        layout.addWidget(grp_search)

        # ── CRS ──
        grp_crs = QGroupBox("Système de coordonnées")
        gc2 = QHBoxLayout(grp_crs)
        gc2.addWidget(QLabel("EPSG :"))
        self.epsg_edit = QLineEdit("2154")
        self.epsg_edit.setMaximumWidth(120)
        gc2.addWidget(self.epsg_edit)
        gc2.addWidget(QLabel("(2154 = Lambert 93 · 32631 = UTM 31N · système projeté requis)"))
        gc2.addStretch()
        layout.addWidget(grp_crs)

        layout.addStretch()
        self.tabs.addTab(tab, "2 — Paramètres")

    # ── Tab 3: Variogramme ────────────────────────────────────────────

    def _build_tab_vario(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.vario_figure = Figure(figsize=(6, 3.5), dpi=100)
        self.vario_canvas = FigureCanvas(self.vario_figure)
        layout.addWidget(self.vario_canvas)

        self.vario_info_label = QLabel("")
        self.vario_info_label.setStyleSheet("font-family: monospace; padding: 6px;")
        self.vario_info_label.setWordWrap(True)
        layout.addWidget(self.vario_info_label)

        self.tabs.addTab(tab, "3 — Variogramme")

    # ── Tab 4: Validation croisée ─────────────────────────────────────

    def _build_tab_crossval(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        self.btn_crossval = QPushButton("⟳  Lancer validation croisée (LOO)")
        self.btn_crossval.setStyleSheet(
            "QPushButton { background-color: #5a8a2c; color: white; font-size: 12px; "
            "font-weight: bold; padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a6a1a; }"
        )
        top_row.addWidget(self.btn_crossval)
        top_row.addSpacing(30)
        self.cv_stats_label = QLabel("Erreur moyenne : —   |   RMSE : —")
        self.cv_stats_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        top_row.addWidget(self.cv_stats_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Splitter: table left, plot right
        splitter = QSplitter(Qt.Horizontal)

        # Table
        self.cv_table = QTableWidget(0, 5)
        self.cv_table.setHorizontalHeaderLabels(["#", "Mesuré", "Estimé", "Erreur", "Err. std."])
        self.cv_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cv_table.setAlternatingRowColors(True)
        self.cv_table.setMinimumWidth(300)
        splitter.addWidget(self.cv_table)

        # Plot
        self.cv_figure = Figure(figsize=(5, 4), dpi=100)
        self.cv_canvas = FigureCanvas(self.cv_figure)
        splitter.addWidget(self.cv_canvas)
        splitter.setSizes([340, 460])

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "4 — Validation croisée")

    # ─────────────────────────────────────
    # Signal handlers
    # ─────────────────────────────────────

    def _on_model_changed(self, model_name):
        is_linear = model_name == "linear"
        manual = self.manual_vario_check.isChecked()
        self._sill_label.setVisible(not is_linear)
        self.sill_spin.setVisible(not is_linear)
        self._range_label.setVisible(not is_linear)
        self.range_spin.setVisible(not is_linear)
        self._slope_label.setVisible(is_linear)
        self.slope_spin.setVisible(is_linear)
        if manual:
            self.sill_spin.setEnabled(not is_linear)
            self.range_spin.setEnabled(not is_linear)
            self.slope_spin.setEnabled(is_linear)

    def _on_manual_toggle(self, checked):
        is_linear = self.model_combo.currentText() == "linear"
        self.nugget_spin.setEnabled(checked)
        self.sill_spin.setEnabled(checked and not is_linear)
        self.range_spin.setEnabled(checked and not is_linear)
        self.slope_spin.setEnabled(checked and is_linear)

    def _on_auto_interval_toggle(self, checked):
        self.contour_interval_spin.setEnabled(not checked)

    def _on_search_mode_changed(self, ell_checked):
        self._ell_frame.setEnabled(ell_checked)

    def _update_spacing_label(self):
        _, coords, _ = self.get_data()
        if len(coords) < 2:
            self._spacing_label.setText("Espacement X : —  |  Y : —")
            return
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        pad = self.pad_spin.value() / 100.0
        span_x = (xmax - xmin) * (1.0 + 2.0 * pad)
        span_y = (ymax - ymin) * (1.0 + 2.0 * pad)
        nx = self.nx_spin.value()
        ny = self.ny_spin.value()
        spx = span_x / (nx - 1) if nx > 1 else 0.0
        spy = span_y / (ny - 1) if ny > 1 else 0.0
        self._spacing_label.setText(
            f"Espacement X : {spx:.4g}  |  Y : {spy:.4g}"
        )

    # ─────────────────────────────────────
    # CSV loading + column mapping
    # ─────────────────────────────────────

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier CSV", "",
            "CSV (*.csv *.txt *.tsv);;Tous (*.*)"
        )
        if path:
            self.csv_path_edit.setText(path)

    def _get_separator(self):
        sep = self.sep_combo.currentText()
        return "\t" if sep == "TAB" else sep

    def _load_csv(self):
        path = self.csv_path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Erreur", "Fichier introuvable.")
            return

        sep = self._get_separator()
        has_header = self.header_check.isChecked()

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f, delimiter=sep)
                rows = [r for r in reader if any(c.strip() for c in r)]
        except Exception as e:
            QMessageBox.critical(self, "Erreur de lecture", str(e))
            return

        if has_header and rows:
            headers = [h.strip() for h in rows[0]]
            self._raw_headers = headers
            self._raw_rows = rows[1:]
        else:
            n_cols = max((len(r) for r in rows), default=4)
            self._raw_headers = [f"col_{i+1}" for i in range(n_cols)]
            self._raw_rows = rows

        self._update_col_mapping(self._raw_headers)
        self._apply_mapping()

    def _update_col_mapping(self, headers):
        """Populate column-mapping combos and auto-detect likely columns."""
        auto_matches = [
            ["ouvrage", "nom", "id", "name", "label", "point"],
            ["x", "lon", "longitude", "est", "easting", "coord_x"],
            ["y", "lat", "latitude", "nord", "northing", "coord_y"],
            ["z", "ngf", "niveau", "piézo", "piezo", "elevation", "altitude", "head"],
        ]
        for cb in self._col_combos:
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(auto)")
            cb.addItems(headers)
            cb.blockSignals(False)

        for role_idx, keywords in enumerate(auto_matches):
            for col_idx, h in enumerate(headers):
                if h.lower().strip() in keywords:
                    self._col_combos[role_idx].setCurrentIndex(col_idx + 1)
                    break

        self._mapping_frame.setVisible(True)

    def _apply_mapping(self):
        """Fill the data table using current combo mapping."""
        if not self._raw_rows:
            return

        n_cols = len(self._raw_headers)
        col_indices = []
        for i, cb in enumerate(self._col_combos):
            idx = cb.currentIndex()
            if idx <= 0:
                col_indices.append(i)  # auto: positional
            else:
                col_indices.append(idx - 1)  # -1 to skip "(auto)"

        self.table.setRowCount(0)
        for row in self._raw_rows:
            if not any(c.strip() for c in row):
                continue
            if not all(ci < len(row) for ci in col_indices):
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c_target, c_source in enumerate(col_indices):
                val = row[c_source].strip() if c_source < len(row) else ""
                self.table.setItem(r, c_target, QTableWidgetItem(val))

        self._update_spacing_label()
        QMessageBox.information(
            self, "Import",
            f"{self.table.rowCount()} ouvrages importés."
        )

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c in range(4):
            self.table.setItem(r, c, QTableWidgetItem(""))

    def _delete_selected(self):
        rows = sorted(set(i.row() for i in self.table.selectedItems()), reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _clear_table(self):
        self.table.setRowCount(0)

    # ─────────────────────────────────────
    # Data accessors
    # ─────────────────────────────────────

    def get_data(self):
        """Return (names, coords, values) from the table."""
        names, coords, values = [], [], []
        for r in range(self.table.rowCount()):
            try:
                name = (self.table.item(r, 0) or QTableWidgetItem("")).text().strip()
                x = float(self.table.item(r, 1).text().replace(",", "."))
                y = float(self.table.item(r, 2).text().replace(",", "."))
                z = float(self.table.item(r, 3).text().replace(",", "."))
            except (ValueError, AttributeError):
                continue
            names.append(name)
            coords.append([x, y])
            values.append(z)
        return names, np.array(coords) if coords else np.empty((0, 2)), np.array(values)

    def get_grid_params(self):
        """Return (nx, ny, pad_percent, nodata_hull)."""
        return (
            self.nx_spin.value(),
            self.ny_spin.value(),
            self.pad_spin.value(),
            self.hull_check.isChecked(),
        )

    def get_variogram_params(self):
        """Return dict with variogram UI state."""
        return {
            "model": self.model_combo.currentText(),
            "n_lags": self.n_lags_spin.value(),
            "lag_size": self.lag_size_spin.value(),  # 0 = auto
            "direction": self.direction_spin.value(),
            "tolerance": self.tolerance_spin.value(),
            "manual": self.manual_vario_check.isChecked(),
            "nugget": self.nugget_spin.value(),
            "sill": self.sill_spin.value(),
            "range": self.range_spin.value(),
            "slope": self.slope_spin.value(),
        }

    def get_search_params(self):
        """Return search neighborhood dict (or None for use-all)."""
        if self.search_all_radio.isChecked():
            return None
        return {
            "use_all": False,
            "radius1": self.r1_spin.value(),
            "radius2": self.r2_spin.value(),
            "angle": self.search_angle_spin.value(),
            "min_neighbors": self.min_nb_spin.value(),
            "max_neighbors": self.max_nb_spin.value(),
        }

    def get_contour_params(self):
        """Return (auto_interval, interval_value, add_labels)."""
        return (
            self.auto_interval_check.isChecked(),
            self.contour_interval_spin.value(),
            self.add_labels_check.isChecked(),
        )

    # ─────────────────────────────────────
    # Variogram plot
    # ─────────────────────────────────────

    def plot_variogram(self, lag_centers, gamma_exp, vario_func, params,
                        n_pairs=None, model_name=""):
        self.vario_figure.clear()
        ax = self.vario_figure.add_subplot(111)

        sizes = None
        if n_pairs is not None and len(n_pairs) > 0:
            sizes = np.clip(np.sqrt(n_pairs) * 20, 20, 200)

        ax.scatter(lag_centers, gamma_exp,
                   s=sizes if sizes is not None else 50,
                   c="#d95f02", edgecolors="k", zorder=5, label="Expérimental")

        if len(lag_centers) > 0:
            h_fit = np.linspace(0, lag_centers.max() * 1.1, 300)
            ax.plot(h_fit, vario_func(h_fit), "-", color="#1b9e77", lw=2,
                    label="Modèle ajusté")

        ax.set_xlabel("Distance h")
        ax.set_ylabel("Semi-variance γ(h)")
        ax.set_title(f"Variogramme ({model_name})" if model_name else "Variogramme")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self.vario_figure.tight_layout()
        self.vario_canvas.draw()

        # Info text
        if params.get("slope") is not None:
            info = (
                f"Modèle : linear  |  "
                f"Nugget (micro-var) = {params['nugget']:.6f}  |  "
                f"Pente = {params['slope']:.8f}"
            )
        else:
            info = (
                f"Modèle : {model_name}  |  "
                f"Nugget (micro-var) = {params['nugget']:.6f}  |  "
                f"Sill = {params['sill']:.6f}  |  "
                f"Range = {params['range']:.4f}"
            )
        if self._loo_rmse is not None:
            info += f"\nLOO RMSE = {self._loo_rmse:.6f}"
            if self._loo_rmsse is not None:
                info += f"   |   RMSSE = {self._loo_rmsse:.4f} (cible ≈ 1)"
        self.vario_info_label.setText(info)

    # ─────────────────────────────────────
    # Cross-validation display
    # ─────────────────────────────────────

    def show_crossval_results(self, cv, names):
        """Populate Tab 4 with LOO results."""
        measured = cv["measured"]
        estimated = cv["estimated"]
        errors = cv["errors"]
        valid = cv["valid"]
        n_failed = cv["n_failed"]

        self._loo_rmse = cv["rmse"]
        self._loo_rmsse = cv["rmsse"]

        def _fmt_stat(v, fmt):
            return "N/A" if not np.isfinite(v) else fmt.format(v)

        stats_line = (
            f"Erreur moyenne : {_fmt_stat(cv['mean_error'], '{:+.4f}')}"
            f"   |   RMSE : {_fmt_stat(cv['rmse'], '{:.4f}')}"
            f"   |   Err. std. moy. : {_fmt_stat(cv['mean_std_error'], '{:+.4f}')}"
            f"   |   RMSSE : {_fmt_stat(cv['rmsse'], '{:.4f}')} (cible ≈ 1)"
        )
        if n_failed:
            stats_line += f"\n⚠ {n_failed} point(s) non résolu(s) par l'ellipse de recherche (exclus des stats)"
        self.cv_stats_label.setText(stats_line)

        # std_err : NaN pour les lignes invalides, valeur réelle pour les valides
        with np.errstate(invalid="ignore"):
            std_err = np.where(valid, errors / np.sqrt(cv["kvar"]), np.nan)

        rmse_valid = cv["rmse"]

        # Table
        n = len(measured)
        self.cv_table.setRowCount(n)
        gray = QColor(150, 150, 150)
        for i in range(n):
            label = names[i] if i < len(names) else str(i + 1)
            self.cv_table.setItem(i, 0, QTableWidgetItem(label))
            self.cv_table.setItem(i, 1, QTableWidgetItem(f"{measured[i]:.4f}"))
            if not valid[i]:
                for col in (2, 3, 4):
                    item = QTableWidgetItem("—")
                    item.setForeground(gray)
                    self.cv_table.setItem(i, col, item)
                continue
            self.cv_table.setItem(i, 2, QTableWidgetItem(f"{estimated[i]:.4f}"))
            err_item = QTableWidgetItem(f"{errors[i]:+.4f}")
            if np.isfinite(rmse_valid) and abs(errors[i]) > rmse_valid:
                err_item.setForeground(QColor(180, 40, 40))
            self.cv_table.setItem(i, 3, err_item)
            std_item = QTableWidgetItem(f"{std_err[i]:+.3f}")
            if abs(std_err[i]) > 2.0:
                std_item.setForeground(QColor(180, 40, 40))
            self.cv_table.setItem(i, 4, std_item)

        # Plot — only finite (measured, estimated) pairs
        self.cv_figure.clear()
        ax = self.cv_figure.add_subplot(111)

        meas_v = measured[valid]
        est_v = estimated[valid]

        if meas_v.size == 0:
            ax.set_title("Validation croisée — aucun point résolu")
            self.cv_figure.tight_layout()
            self.cv_canvas.draw()
            self.tabs.setCurrentIndex(3)
            return

        all_vals = np.concatenate([meas_v, est_v])
        lo, hi = all_vals.min(), all_vals.max()
        margin = max((hi - lo) * 0.05, 1e-6)
        lim = (lo - margin, hi + margin)

        ax.plot(lim, lim, "--", color="#888", lw=1, label="Référence (y=x)")

        if meas_v.size >= 2:
            slope_bf, intercept_bf = np.polyfit(meas_v, est_v, 1)
            x_bf = np.array(lim)
            ax.plot(x_bf, slope_bf * x_bf + intercept_bf,
                    "-", color="#d62728", lw=1.5, alpha=0.8,
                    label=f"Best fit (a={slope_bf:.3f})")

        ax.scatter(meas_v, est_v, c="#1f77b4", edgecolors="k", s=60, zorder=5)

        valid_indices = np.where(valid)[0]
        for vi, i in enumerate(valid_indices):
            lbl = names[i] if i < len(names) else str(i + 1)
            ax.annotate(lbl, (meas_v[vi], est_v[vi]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Mesuré")
        ax.set_ylabel("Estimé (LOO)")
        title = "Validation croisée Leave-One-Out"
        if n_failed:
            title += f" ({n_failed} point(s) exclus)"
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        self.cv_figure.tight_layout()
        self.cv_canvas.draw()

        self.tabs.setCurrentIndex(3)
