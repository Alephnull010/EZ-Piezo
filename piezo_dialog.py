"""
piezo_dialog.py - Qt dialog for EZ Piezo plugin.
"""

import os
import csv

import numpy as np

from qgis.PyQt.QtCore import Qt, QSize
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QTabWidget, QWidget, QProgressBar,
    QHeaderView, QCheckBox, QSplitter, QFrame, QRadioButton,
    QButtonGroup,
)
from qgis.PyQt.QtGui import QColor, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _flag_icon(flag):
    """Load a flag icon from the plugin's icons/ directory.
    flag: 'fr' or 'gb'.
    """
    path = os.path.join(os.path.dirname(__file__), 'icons', f'{flag}.png')
    return QIcon(path)


_STRINGS = {
    'fr': {
        'window_title': "EZ Piezo — Carte piézométrique par Kriging",
        'subtitle': "Interpolation piézométrique par Ordinary Kriging",
        'tab_data': "1 — Données",
        'tab_params': "2 — Paramètres",
        'tab_vario': "3 — Variogramme",
        'tab_crossval': "4 — Validation croisée",
        'btn_run': "▶  Lancer le Kriging",
        'lang_btn': "🇬🇧",
        # Tab 1
        'csv_placeholder': "Chemin vers fichier CSV …",
        'btn_browse': "Parcourir…",
        'lbl_sep': "Séparateur :",
        'header_check': "Première ligne = en-tête",
        'lbl_mapping': "Mapping :",
        'col_labels': ["Ouvrage", "X", "Y", "Z (m)"],
        'btn_apply_mapping': "Appliquer",
        'table_headers': ["Ouvrage", "X", "Y", "Z (m NGF)"],
        'btn_add': "+ Ajouter ligne",
        'btn_del': "− Supprimer sélection",
        'btn_clear': "Tout effacer",
        # Tab 2 — Variogram
        'grp_vario': "Variogramme",
        'lbl_model': "Modèle :",
        'lbl_nlags': "Nb lags :",
        'lbl_lagsize': "Lag size (0=auto) :",
        'lbl_minpairs': "Min paires/lag :",
        'min_pairs_none': "aucun",
        'lbl_direction': "Direction (°, depuis Est) :",
        'lbl_tolerance': "Tolérance (°) :",
        'direction_tooltip': "0° = Est, 90° = Nord. Ignoré si tolérance = 90°.",
        'tolerance_tooltip': "90° = omnidirectionnel (comportement par défaut).",
        'manual_check': "Paramètres manuels",
        'lbl_slope': "Pente :",
        # Tab 2 — Grid
        'grp_grid': "Grille d'interpolation",
        'lbl_nx': "Noeuds X :",
        'lbl_ny': "Noeuds Y :",
        'lbl_margin': "Marge (%) :",
        'hull_check': "NoData hors convex hull",
        'hull_tooltip': "Marque comme NoData les noeuds de grille en dehors de l'enveloppe convexe des points.",
        'spacing_empty': "Espacement X : —  |  Y : —",
        'spacing_fmt': "Espacement X : {sx}  |  Y : {sy}",
        # Tab 2 — Contours
        'grp_contour': "Courbes isopièzes",
        'auto_interval': "Intervalle auto",
        'lbl_interval': "Intervalle (m) :",
        'add_labels': "Étiquettes sur isopièzes",
        'lbl_major': "Principale toutes les :",
        'major_suffix': " iso.",
        'lbl_offset': "Décalage maîtresse :",
        'btn_restyle': "↺ Réappliquer style isopièzes",
        # Tab 2 — Search
        'grp_search': "Voisinage de recherche",
        'search_all': "Toutes les données",
        'search_ell': "Ellipse de recherche",
        'lbl_r1': "Rayon 1 :",
        'lbl_r2': "Rayon 2 :",
        'lbl_angle': "Angle (°) :",
        'lbl_min_nb': "Min données :",
        'lbl_max_nb': "Max données :",
        # Tab 2 — CRS
        'grp_crs': "Système de coordonnées",
        'lbl_crs_hint': "(2154 = Lambert 93 · 32631 = UTM 31N · système projeté requis)",
        # Tab 2 — Flow
        'grp_flow': "Vecteurs de flux",
        'flow_check': "Afficher les vecteurs de flux",
        'lbl_step_x': "Pas X :",
        'lbl_step_y': "Pas Y :",
        'btn_refresh_flow': "↺ Rafraîchir les vecteurs",
        # Tab 4
        'btn_crossval': "⟳  Relancer validation croisée (LOO)",
        'cv_stats_init': "Erreur moyenne : —   |   RMSE : —",
        'cv_headers': ["#", "Mesuré", "Estimé", "Erreur", "Err. std."],
        'cv_stats_fmt': (
            "Erreur moyenne : {me}   |   RMSE : {rmse}"
            "   |   Err. std. moy. : {mse}   |   RMSSE : {rmsse} (cible ≈ 1)"
        ),
        'cv_unresolved': "⚠ {n} point(s) non résolu(s) par l'ellipse de recherche (exclus des stats)",
        'cv_plot_no_pts': "Validation croisée — aucun point résolu",
        'cv_plot_title': "Validation croisée Leave-One-Out",
        'cv_plot_excluded': "({n} point(s) exclus)",
        'cv_ref_label': "Référence (y=x)",
        'cv_xlabel': "Estimé (LOO)",
        'cv_ylabel': "Mesuré",
        # Variogram plot
        'vario_exp_label': "Expérimental",
        'vario_fit_label': "Modèle ajusté",
        'vario_xlabel': "Distance h",
        'vario_ylabel': "Semi-variance γ(h)",
        'vario_title': "Variogramme",
        'vario_info_linear': "Modèle : linear  |  Nugget = {nugget:.6f}  |  Pente = {slope:.8f}",
        'vario_info_fmt': "Modèle : {model}  |  Nugget = {nugget:.6f}  |  Sill = {sill:.6f}  |  Range = {range:.4f}",
        'vario_rmsse_target': "(cible ≈ 1)",
        # File dialogs / messages
        'dlg_open_csv': "Ouvrir un fichier CSV",
        'dlg_csv_filter': "CSV (*.csv *.txt *.tsv);;Tous (*.*)",
        'err_file_not_found': "Fichier introuvable.",
        'err_read_title': "Erreur de lecture",
        'import_title': "Import",
        'import_success': "{n} ouvrages importés.",
    },
    'en': {
        'window_title': "EZ Piezo — Piezometric Map by Kriging",
        'subtitle': "Piezometric interpolation by Ordinary Kriging",
        'tab_data': "1 — Data",
        'tab_params': "2 — Parameters",
        'tab_vario': "3 — Variogram",
        'tab_crossval': "4 — Cross-validation",
        'btn_run': "▶  Run Kriging",
        'lang_btn': "🇫🇷",
        # Tab 1
        'csv_placeholder': "Path to CSV file …",
        'btn_browse': "Browse…",
        'lbl_sep': "Separator:",
        'header_check': "First row = header",
        'lbl_mapping': "Mapping:",
        'col_labels': ["Well", "X", "Y", "Z (m)"],
        'btn_apply_mapping': "Apply",
        'table_headers': ["Well", "X", "Y", "Z (m NGF)"],
        'btn_add': "+ Add row",
        'btn_del': "− Delete selection",
        'btn_clear': "Clear all",
        # Tab 2 — Variogram
        'grp_vario': "Variogram",
        'lbl_model': "Model:",
        'lbl_nlags': "Nb lags:",
        'lbl_lagsize': "Lag size (0=auto):",
        'lbl_minpairs': "Min pairs/lag:",
        'min_pairs_none': "none",
        'lbl_direction': "Direction (°, from East):",
        'lbl_tolerance': "Tolerance (°):",
        'direction_tooltip': "0° = East, 90° = North. Ignored if tolerance = 90°.",
        'tolerance_tooltip': "90° = omnidirectional (default behaviour).",
        'manual_check': "Manual parameters",
        'lbl_slope': "Slope:",
        # Tab 2 — Grid
        'grp_grid': "Interpolation grid",
        'lbl_nx': "X nodes:",
        'lbl_ny': "Y nodes:",
        'lbl_margin': "Padding (%):",
        'hull_check': "NoData outside convex hull",
        'hull_tooltip': "Set grid nodes outside the convex hull of input points to NoData.",
        'spacing_empty': "Spacing X: —  |  Y: —",
        'spacing_fmt': "Spacing X: {sx}  |  Y: {sy}",
        # Tab 2 — Contours
        'grp_contour': "Contour lines",
        'auto_interval': "Auto interval",
        'lbl_interval': "Interval (m):",
        'add_labels': "Labels on contours",
        'lbl_major': "Major every:",
        'major_suffix': " lines",
        'lbl_offset': "Major offset:",
        'btn_restyle': "↺ Re-apply contour style",
        # Tab 2 — Search
        'grp_search': "Search neighborhood",
        'search_all': "All data",
        'search_ell': "Search ellipse",
        'lbl_r1': "Radius 1:",
        'lbl_r2': "Radius 2:",
        'lbl_angle': "Angle (°):",
        'lbl_min_nb': "Min data:",
        'lbl_max_nb': "Max data:",
        # Tab 2 — CRS
        'grp_crs': "Coordinate system",
        'lbl_crs_hint': "(2154 = Lambert 93 · 32631 = UTM 31N · projected CRS required)",
        # Tab 2 — Flow
        'grp_flow': "Flow vectors",
        'flow_check': "Show flow vectors",
        'lbl_step_x': "Step X:",
        'lbl_step_y': "Step Y:",
        'btn_refresh_flow': "↺ Refresh vectors",
        # Tab 4
        'btn_crossval': "⟳  Re-run cross-validation (LOO)",
        'cv_stats_init': "Mean error: —   |   RMSE: —",
        'cv_headers': ["#", "Measured", "Estimated", "Error", "Std. err."],
        'cv_stats_fmt': (
            "Mean error: {me}   |   RMSE: {rmse}"
            "   |   Mean std. err.: {mse}   |   RMSSE: {rmsse} (target ≈ 1)"
        ),
        'cv_unresolved': "⚠ {n} point(s) unresolved by search ellipse (excluded from stats)",
        'cv_plot_no_pts': "Cross-validation — no resolved points",
        'cv_plot_title': "Leave-One-Out cross-validation",
        'cv_plot_excluded': "({n} point(s) excluded)",
        'cv_ref_label': "Reference (y=x)",
        'cv_xlabel': "Estimated (LOO)",
        'cv_ylabel': "Measured",
        # Variogram plot
        'vario_exp_label': "Experimental",
        'vario_fit_label': "Fitted model",
        'vario_xlabel': "Distance h",
        'vario_ylabel': "Semi-variance γ(h)",
        'vario_title': "Variogram",
        'vario_info_linear': "Model: linear  |  Nugget = {nugget:.6f}  |  Slope = {slope:.8f}",
        'vario_info_fmt': "Model: {model}  |  Nugget = {nugget:.6f}  |  Sill = {sill:.6f}  |  Range = {range:.4f}",
        'vario_rmsse_target': "(target ≈ 1)",
        # File dialogs / messages
        'dlg_open_csv': "Open CSV file",
        'dlg_csv_filter': "CSV (*.csv *.txt *.tsv);;All (*.*)",
        'err_file_not_found': "File not found.",
        'err_read_title': "Read error",
        'import_title': "Import",
        'import_success': "{n} wells imported.",
    },
}


class PiezoKrigingDialog(QDialog):
    """Main dialog for PiezoKriging plugin."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1000, 720)
        self.resize(1100, 800)

        # State for column mapping
        self._raw_rows = []
        self._raw_headers = []
        # State for LOO stats (updated after cross-validation)
        self._loo_rmse = None
        self._loo_rmsse = None
        self._lang = 'fr'

        self._build_ui()
        self._apply_language('fr')

    # ─────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Header: title/subtitle on the left, language toggle on the right
        hdr_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel("EZ Piezo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c5f8a;")
        self._subtitle_label = QLabel()
        self._subtitle_label.setStyleSheet("font-size: 11px; color: #666; margin-bottom: 8px;")
        title_col.addWidget(title)
        title_col.addWidget(self._subtitle_label)
        hdr_row.addLayout(title_col)
        hdr_row.addStretch()
        self._btn_lang = QPushButton()
        self._btn_lang.setFixedSize(40, 28)
        self._btn_lang.setIconSize(QSize(30, 20))
        self._btn_lang.setStyleSheet("padding: 2px; border: 1px solid #bbb; border-radius: 3px;")
        self._btn_lang.clicked.connect(self._toggle_lang)
        hdr_row.addWidget(self._btn_lang, 0, Qt.AlignTop)
        main_layout.addLayout(hdr_row)

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
        self.btn_run = QPushButton()
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #2c7fb8; color: white; font-size: 14px; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1a5a8a; }"
        )
        run_row.addStretch()
        run_row.addWidget(self.btn_run)
        run_row.addStretch()
        main_layout.addLayout(run_row)

    # ── Tab 1: Data ───────────────────────────────────────────────────

    def _build_tab_data(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File row
        file_row = QHBoxLayout()
        self.csv_path_edit = QLineEdit()
        self._btn_browse = QPushButton()
        self._btn_browse.clicked.connect(self._browse_csv)
        file_row.addWidget(self.csv_path_edit, 4)
        file_row.addWidget(self._btn_browse, 1)
        layout.addLayout(file_row)

        # Separator + header
        sep_row = QHBoxLayout()
        self._lbl_sep = QLabel()
        sep_row.addWidget(self._lbl_sep)
        self.sep_combo = QComboBox()
        self.sep_combo.addItems([";", ",", "TAB", "|"])
        sep_row.addWidget(self.sep_combo)
        sep_row.addSpacing(20)
        self.header_check = QCheckBox()
        self.header_check.setChecked(True)
        sep_row.addWidget(self.header_check)
        sep_row.addStretch()
        layout.addLayout(sep_row)

        # Column mapping (shown after load)
        self._mapping_frame = QFrame()
        self._mapping_frame.setFrameShape(QFrame.StyledPanel)
        map_layout = QHBoxLayout(self._mapping_frame)
        self._lbl_mapping = QLabel()
        map_layout.addWidget(self._lbl_mapping)
        self._col_combos = []
        self._col_label_widgets = []
        for _ in range(4):
            col_lbl = QLabel()
            self._col_label_widgets.append(col_lbl)
            map_layout.addWidget(col_lbl)
            cb = QComboBox()
            cb.setMinimumWidth(120)
            cb.addItem("(auto)")
            self._col_combos.append(cb)
            map_layout.addWidget(cb)
        self._btn_apply_mapping = QPushButton()
        self._btn_apply_mapping.clicked.connect(self._apply_mapping)
        map_layout.addWidget(self._btn_apply_mapping)
        map_layout.addStretch()
        self._mapping_frame.setVisible(False)
        layout.addWidget(self._mapping_frame)

        # Data table
        self.table = QTableWidget(0, 4)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Add / remove buttons
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton()
        self._btn_add.clicked.connect(self._add_row)
        self._btn_del = QPushButton()
        self._btn_del.clicked.connect(self._delete_selected)
        self._btn_clear = QPushButton()
        self._btn_clear.clicked.connect(self._clear_table)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_del)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "")

    # ── Tab 2: Parameters ─────────────────────────────────────────────

    def _build_tab_params(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Variogramme ──
        self._grp_vario = QGroupBox()
        gv = QVBoxLayout(self._grp_vario)

        row1 = QHBoxLayout()
        self._lbl_model = QLabel()
        row1.addWidget(self._lbl_model)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["linear", "spherical", "exponential", "gaussian"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row1.addWidget(self.model_combo)
        self.force_nugget_zero_check = QCheckBox("Nugget = 0")
        self.force_nugget_zero_check.setVisible(False)  # shown only when linear is selected
        row1.addWidget(self.force_nugget_zero_check)
        row1.addSpacing(12)
        self._lbl_nlags = QLabel()
        row1.addWidget(self._lbl_nlags)
        self.n_lags_spin = QSpinBox()
        self.n_lags_spin.setRange(5, 100)
        self.n_lags_spin.setValue(15)
        row1.addWidget(self.n_lags_spin)
        row1.addSpacing(12)
        self._lbl_lagsize = QLabel()
        row1.addWidget(self._lbl_lagsize)
        self.lag_size_spin = QDoubleSpinBox()
        self.lag_size_spin.setRange(0.0, 9999999.0)
        self.lag_size_spin.setDecimals(4)
        self.lag_size_spin.setValue(0.0)
        self.lag_size_spin.setSpecialValueText("auto")
        row1.addWidget(self.lag_size_spin)
        row1.addSpacing(12)
        self._lbl_minpairs = QLabel()
        row1.addWidget(self._lbl_minpairs)
        self.min_pairs_spin = QSpinBox()
        self.min_pairs_spin.setRange(0, 100)
        self.min_pairs_spin.setValue(0)
        self.min_pairs_spin.setMaximumWidth(60)
        self.min_pairs_spin.setSpecialValueText("aucun")
        row1.addWidget(self.min_pairs_spin)
        row1.addStretch()
        gv.addLayout(row1)

        row_dir = QHBoxLayout()
        self._lbl_direction = QLabel()
        row_dir.addWidget(self._lbl_direction)
        self.direction_spin = QDoubleSpinBox()
        self.direction_spin.setRange(-180.0, 180.0)
        self.direction_spin.setValue(0.0)
        self.direction_spin.setDecimals(1)
        row_dir.addWidget(self.direction_spin)
        row_dir.addSpacing(12)
        self._lbl_tolerance = QLabel()
        row_dir.addWidget(self._lbl_tolerance)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.1, 90.0)
        self.tolerance_spin.setValue(90.0)
        self.tolerance_spin.setDecimals(1)
        row_dir.addWidget(self.tolerance_spin)
        row_dir.addStretch()
        gv.addLayout(row_dir)

        # Manual params row
        row_manual = QHBoxLayout()
        self.manual_vario_check = QCheckBox()
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

        self._slope_label = QLabel()
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
        layout.addWidget(self._grp_vario)

        # ── Grille ──
        self._grp_grid = QGroupBox()
        gg = QVBoxLayout(self._grp_grid)

        row_g1 = QHBoxLayout()
        self._lbl_nx = QLabel()
        row_g1.addWidget(self._lbl_nx)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 2000)
        self.nx_spin.setValue(100)
        self.nx_spin.valueChanged.connect(self._update_spacing_label)
        row_g1.addWidget(self.nx_spin)
        row_g1.addSpacing(12)
        self._lbl_ny = QLabel()
        row_g1.addWidget(self._lbl_ny)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 2000)
        self.ny_spin.setValue(100)
        self.ny_spin.valueChanged.connect(self._update_spacing_label)
        row_g1.addWidget(self.ny_spin)
        row_g1.addSpacing(12)
        self._lbl_margin = QLabel()
        row_g1.addWidget(self._lbl_margin)
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
        self.hull_check = QCheckBox()
        row_g2.addWidget(self.hull_check)
        gg.addLayout(row_g2)

        layout.addWidget(self._grp_grid)

        # ── Contours ──
        self._grp_contour = QGroupBox()
        gc = QHBoxLayout(self._grp_contour)
        self.auto_interval_check = QCheckBox()
        self.auto_interval_check.setChecked(True)
        self.auto_interval_check.toggled.connect(self._on_auto_interval_toggle)
        gc.addWidget(self.auto_interval_check)
        self._lbl_interval = QLabel()
        gc.addWidget(self._lbl_interval)
        self.contour_interval_spin = QDoubleSpinBox()
        self.contour_interval_spin.setRange(0.0001, 10000.0)
        self.contour_interval_spin.setValue(1.0)
        self.contour_interval_spin.setDecimals(4)
        self.contour_interval_spin.setEnabled(False)
        gc.addWidget(self.contour_interval_spin)
        gc.addSpacing(20)
        self.add_labels_check = QCheckBox()
        self.add_labels_check.setChecked(True)
        gc.addWidget(self.add_labels_check)
        gc.addSpacing(10)
        self._lbl_major_nth = QLabel()
        gc.addWidget(self._lbl_major_nth)
        self.major_nth_spin = QSpinBox()
        self.major_nth_spin.setRange(2, 20)
        self.major_nth_spin.setValue(5)
        gc.addWidget(self.major_nth_spin)
        self._lbl_major_offset = QLabel()
        gc.addWidget(self._lbl_major_offset)
        self.major_offset_spin = QDoubleSpinBox()
        self.major_offset_spin.setRange(-9999.0, 9999.0)
        self.major_offset_spin.setValue(0.0)
        self.major_offset_spin.setDecimals(2)
        self.major_offset_spin.setSingleStep(0.5)
        gc.addWidget(self.major_offset_spin)
        self.btn_restyle_contours = QPushButton()
        gc.addWidget(self.btn_restyle_contours)
        gc.addStretch()
        layout.addWidget(self._grp_contour)

        # ── Voisinage de recherche ──
        self._grp_search = QGroupBox()
        gs = QVBoxLayout(self._grp_search)

        self._search_btn_group = QButtonGroup(self)
        self.search_all_radio = QRadioButton()
        self.search_ell_radio = QRadioButton()
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

        self._lbl_r1 = QLabel()
        ell_layout.addWidget(self._lbl_r1)
        self.r1_spin = QDoubleSpinBox()
        self.r1_spin.setRange(0.001, 9999999.0)
        self.r1_spin.setDecimals(2)
        self.r1_spin.setValue(1000.0)
        ell_layout.addWidget(self.r1_spin)

        ell_layout.addSpacing(8)
        self._lbl_r2 = QLabel()
        ell_layout.addWidget(self._lbl_r2)
        self.r2_spin = QDoubleSpinBox()
        self.r2_spin.setRange(0.001, 9999999.0)
        self.r2_spin.setDecimals(2)
        self.r2_spin.setValue(1000.0)
        ell_layout.addWidget(self.r2_spin)

        ell_layout.addSpacing(8)
        self._lbl_angle = QLabel()
        ell_layout.addWidget(self._lbl_angle)
        self.search_angle_spin = QDoubleSpinBox()
        self.search_angle_spin.setRange(-180.0, 180.0)
        self.search_angle_spin.setDecimals(1)
        self.search_angle_spin.setValue(0.0)
        ell_layout.addWidget(self.search_angle_spin)

        ell_layout.addSpacing(8)
        self._lbl_min_nb = QLabel()
        ell_layout.addWidget(self._lbl_min_nb)
        self.min_nb_spin = QSpinBox()
        self.min_nb_spin.setRange(1, 999)
        self.min_nb_spin.setValue(1)
        ell_layout.addWidget(self.min_nb_spin)

        ell_layout.addSpacing(8)
        self._lbl_max_nb = QLabel()
        ell_layout.addWidget(self._lbl_max_nb)
        self.max_nb_spin = QSpinBox()
        self.max_nb_spin.setRange(1, 9999)
        self.max_nb_spin.setValue(20)
        ell_layout.addWidget(self.max_nb_spin)

        ell_layout.addStretch()
        self._ell_frame.setEnabled(False)
        gs.addWidget(self._ell_frame)
        layout.addWidget(self._grp_search)

        # ── CRS ──
        self._grp_crs = QGroupBox()
        gc2 = QHBoxLayout(self._grp_crs)
        gc2.addWidget(QLabel("EPSG :"))
        self.epsg_edit = QLineEdit("2154")
        self.epsg_edit.setMaximumWidth(120)
        gc2.addWidget(self.epsg_edit)
        self._lbl_crs_hint = QLabel()
        gc2.addWidget(self._lbl_crs_hint)
        gc2.addStretch()
        layout.addWidget(self._grp_crs)

        # ── Flow vectors ──
        self._grp_flow = QGroupBox()
        gf = QHBoxLayout(self._grp_flow)
        self.flow_vectors_check = QCheckBox()
        self.flow_vectors_check.setChecked(True)
        gf.addWidget(self.flow_vectors_check)
        gf.addSpacing(16)
        self._lbl_step_x = QLabel()
        gf.addWidget(self._lbl_step_x)
        self.flow_step_x_spin = QSpinBox()
        self.flow_step_x_spin.setRange(2, 100)
        self.flow_step_x_spin.setValue(20)
        self.flow_step_x_spin.setMaximumWidth(60)
        gf.addWidget(self.flow_step_x_spin)
        gf.addSpacing(8)
        self._lbl_step_y = QLabel()
        gf.addWidget(self._lbl_step_y)
        self.flow_step_y_spin = QSpinBox()
        self.flow_step_y_spin.setRange(2, 100)
        self.flow_step_y_spin.setValue(20)
        self.flow_step_y_spin.setMaximumWidth(60)
        gf.addWidget(self.flow_step_y_spin)
        gf.addSpacing(12)
        self.btn_refresh_flow = QPushButton()
        self.btn_refresh_flow.setEnabled(False)
        gf.addWidget(self.btn_refresh_flow)
        gf.addStretch()
        self.flow_vectors_check.toggled.connect(self.flow_step_x_spin.setEnabled)
        self.flow_vectors_check.toggled.connect(self.flow_step_y_spin.setEnabled)
        layout.addWidget(self._grp_flow)

        layout.addStretch()
        self.tabs.addTab(tab, "")

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

        self.tabs.addTab(tab, "")

    # ── Tab 4: Cross-validation ───────────────────────────────────────

    def _build_tab_crossval(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        self.btn_crossval = QPushButton()
        self.btn_crossval.setStyleSheet(
            "QPushButton { background-color: #5a8a2c; color: white; font-size: 12px; "
            "font-weight: bold; padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a6a1a; }"
        )
        top_row.addWidget(self.btn_crossval)
        top_row.addSpacing(30)
        self.cv_stats_label = QLabel()
        self.cv_stats_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        top_row.addWidget(self.cv_stats_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Splitter: table left, plot right
        splitter = QSplitter(Qt.Horizontal)

        # Table
        self.cv_table = QTableWidget(0, 5)
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
        self.tabs.addTab(tab, "")

    # ─────────────────────────────────────
    # Signal handlers
    # ─────────────────────────────────────

    def _on_model_changed(self, model_name):
        is_linear = model_name == "linear"
        manual = self.manual_vario_check.isChecked()
        self.force_nugget_zero_check.setVisible(is_linear)
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
        s = _STRINGS[self._lang]
        _, coords, _ = self.get_data()
        if len(coords) < 2:
            self._spacing_label.setText(s['spacing_empty'])
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
        self._spacing_label.setText(s['spacing_fmt'].format(sx=f"{spx:.4g}", sy=f"{spy:.4g}"))

    # ─────────────────────────────────────
    # CSV loading + column mapping
    # ─────────────────────────────────────

    def _browse_csv(self):
        s = _STRINGS[self._lang]
        path, _ = QFileDialog.getOpenFileName(
            self, s['dlg_open_csv'], "", s['dlg_csv_filter']
        )
        if path:
            self.csv_path_edit.setText(path)
            self._load_csv()

    def _get_separator(self):
        sep = self.sep_combo.currentText()
        return "\t" if sep == "TAB" else sep

    def _load_csv(self):
        s = _STRINGS[self._lang]
        path = self.csv_path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Erreur", s['err_file_not_found'])
            return

        sep = self._get_separator()
        has_header = self.header_check.isChecked()

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f, delimiter=sep)
                rows = [r for r in reader if any(c.strip() for c in r)]
        except Exception as e:
            QMessageBox.critical(self, s['err_read_title'], str(e))
            return

        if has_header and rows:
            headers = [h.strip() for h in rows[0]]
            self._raw_headers = headers
            self._raw_rows = rows[1:]
        else:
            n_cols = max((len(r) for r in rows), default=4)
            self._raw_headers = [f"col_{i + 1}" for i in range(n_cols)]
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
        s = _STRINGS[self._lang]
        QMessageBox.information(
            self, s['import_title'],
            s['import_success'].format(n=self.table.rowCount())
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
            "lag_size": self.lag_size_spin.value(),    # 0 = auto
            "min_pairs": self.min_pairs_spin.value(),
            "force_nugget_zero": self.force_nugget_zero_check.isChecked(),
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
        """Return (auto_interval, interval_value, add_labels, major_nth, major_offset)."""
        return (
            self.auto_interval_check.isChecked(),
            self.contour_interval_spin.value(),
            self.add_labels_check.isChecked(),
            self.major_nth_spin.value(),
            self.major_offset_spin.value(),
        )

    def get_flow_params(self):
        """Return flow vector display parameters."""
        return {
            "enabled": self.flow_vectors_check.isChecked(),
            "step_x": self.flow_step_x_spin.value(),
            "step_y": self.flow_step_y_spin.value(),
        }

    # ─────────────────────────────────────
    # Variogram plot
    # ─────────────────────────────────────

    def plot_variogram(
            self, lag_centers, gamma_exp, vario_func, params,
            n_pairs=None, model_name=""):
        self.vario_figure.clear()
        ax = self.vario_figure.add_subplot(111)

        sizes = None
        if n_pairs is not None and len(n_pairs) > 0:
            sizes = np.clip(np.sqrt(n_pairs) * 20, 20, 200)

        s = _STRINGS[self._lang]
        ax.scatter(lag_centers, gamma_exp,
                   s=sizes if sizes is not None else 50,
                   c="#d95f02", edgecolors="k", zorder=5, label=s['vario_exp_label'])

        if len(lag_centers) > 0:
            h_fit = np.linspace(0, lag_centers.max() * 1.1, 300)
            ax.plot(h_fit, vario_func(h_fit), "-", color="#1b9e77", lw=2,
                    label=s['vario_fit_label'])

        ax.set_xlabel(s['vario_xlabel'])
        ax.set_ylabel(s['vario_ylabel'])
        title = s['vario_title']
        ax.set_title(f"{title} ({model_name})" if model_name else title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self.vario_figure.tight_layout()
        self.vario_canvas.draw()

        # Info text
        if params.get("slope") is not None:
            info = s['vario_info_linear'].format(
                nugget=params['nugget'], slope=params['slope']
            )
        else:
            info = s['vario_info_fmt'].format(
                model=model_name, nugget=params['nugget'],
                sill=params['sill'], range=params['range']
            )
        if self._loo_rmse is not None:
            info += f"\nLOO RMSE = {self._loo_rmse:.6f}"
            if self._loo_rmsse is not None:
                info += f"   |   RMSSE = {self._loo_rmsse:.4f} {s['vario_rmsse_target']}"
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

        s = _STRINGS[self._lang]

        def _fmt_stat(v, fmt):
            return "N/A" if not np.isfinite(v) else fmt.format(v)

        stats_line = s['cv_stats_fmt'].format(
            me=_fmt_stat(cv['mean_error'], '{:+.4f}'),
            rmse=_fmt_stat(cv['rmse'], '{:.4f}'),
            mse=_fmt_stat(cv['mean_std_error'], '{:+.4f}'),
            rmsse=_fmt_stat(cv['rmsse'], '{:.4f}'),
        )
        if n_failed:
            stats_line += "\n" + s['cv_unresolved'].format(n=n_failed)
        self.cv_stats_label.setText(stats_line)

        # std_err: NaN for unresolved rows, actual value for valid ones
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
            ax.set_title(s['cv_plot_no_pts'])
            self.cv_figure.tight_layout()
            self.cv_canvas.draw()
            self.tabs.setCurrentIndex(3)
            return

        all_vals = np.concatenate([meas_v, est_v])
        lo, hi = all_vals.min(), all_vals.max()
        margin = max((hi - lo) * 0.05, 1e-6)
        lim = (lo - margin, hi + margin)

        ax.plot(lim, lim, "--", color="#888", lw=1, label=s['cv_ref_label'])

        if meas_v.size >= 2:
            slope_bf, intercept_bf = np.polyfit(est_v, meas_v, 1)
            x_bf = np.array(lim)
            ax.plot(x_bf, slope_bf * x_bf + intercept_bf,
                    "-", color="#d62728", lw=1.5, alpha=0.8,
                    label=f"Best fit (a={slope_bf:.3f})")

        ax.scatter(est_v, meas_v, c="#1f77b4", edgecolors="k", s=60, zorder=5)

        valid_indices = np.where(valid)[0]
        for vi, i in enumerate(valid_indices):
            lbl = names[i] if i < len(names) else str(i + 1)
            ax.annotate(lbl, (est_v[vi], meas_v[vi]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(s['cv_xlabel'])
        ax.set_ylabel(s['cv_ylabel'])
        cv_title = s['cv_plot_title']
        if n_failed:
            cv_title += " " + s['cv_plot_excluded'].format(n=n_failed)
        ax.set_title(cv_title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        self.cv_figure.tight_layout()
        self.cv_canvas.draw()

        self.tabs.setCurrentIndex(3)

    # ─────────────────────────────────────
    # Language toggle
    # ─────────────────────────────────────

    def _toggle_lang(self):
        self._apply_language('en' if self._lang == 'fr' else 'fr')

    def _apply_language(self, lang):
        self._lang = lang
        s = _STRINGS[lang]

        # Show the OTHER language's flag (the one you'd switch to)
        self._btn_lang.setIcon(_flag_icon('uk' if lang == 'fr' else 'fr'))
        self.setWindowTitle(s['window_title'])
        self._subtitle_label.setText(s['subtitle'])
        self.btn_run.setText(s['btn_run'])

        self.tabs.setTabText(0, s['tab_data'])
        self.tabs.setTabText(1, s['tab_params'])
        self.tabs.setTabText(2, s['tab_vario'])
        self.tabs.setTabText(3, s['tab_crossval'])

        # Tab 1
        self.csv_path_edit.setPlaceholderText(s['csv_placeholder'])
        self._btn_browse.setText(s['btn_browse'])
        self._lbl_sep.setText(s['lbl_sep'])
        self.header_check.setText(s['header_check'])
        self._lbl_mapping.setText(s['lbl_mapping'])
        for widget, col_lbl in zip(self._col_label_widgets, s['col_labels']):
            widget.setText(f"{col_lbl} →")
        self._btn_apply_mapping.setText(s['btn_apply_mapping'])
        self.table.setHorizontalHeaderLabels(s['table_headers'])
        self._btn_add.setText(s['btn_add'])
        self._btn_del.setText(s['btn_del'])
        self._btn_clear.setText(s['btn_clear'])

        # Tab 2 — Variogram
        self._grp_vario.setTitle(s['grp_vario'])
        self._lbl_model.setText(s['lbl_model'])
        self._lbl_nlags.setText(s['lbl_nlags'])
        self._lbl_lagsize.setText(s['lbl_lagsize'])
        self._lbl_minpairs.setText(s['lbl_minpairs'])
        self.min_pairs_spin.setSpecialValueText(s['min_pairs_none'])
        self._lbl_direction.setText(s['lbl_direction'])
        self._lbl_tolerance.setText(s['lbl_tolerance'])
        self.direction_spin.setToolTip(s['direction_tooltip'])
        self.tolerance_spin.setToolTip(s['tolerance_tooltip'])
        self.manual_vario_check.setText(s['manual_check'])
        self._slope_label.setText(s['lbl_slope'])

        # Tab 2 — Grid
        self._grp_grid.setTitle(s['grp_grid'])
        self._lbl_nx.setText(s['lbl_nx'])
        self._lbl_ny.setText(s['lbl_ny'])
        self._lbl_margin.setText(s['lbl_margin'])
        self.hull_check.setText(s['hull_check'])
        self.hull_check.setToolTip(s['hull_tooltip'])
        self._update_spacing_label()

        # Tab 2 — Contours
        self._grp_contour.setTitle(s['grp_contour'])
        self.auto_interval_check.setText(s['auto_interval'])
        self._lbl_interval.setText(s['lbl_interval'])
        self.add_labels_check.setText(s['add_labels'])
        self._lbl_major_nth.setText(s['lbl_major'])
        self.major_nth_spin.setSuffix(s['major_suffix'])
        self._lbl_major_offset.setText(s['lbl_offset'])
        self.btn_restyle_contours.setText(s['btn_restyle'])

        # Tab 2 — Search
        self._grp_search.setTitle(s['grp_search'])
        self.search_all_radio.setText(s['search_all'])
        self.search_ell_radio.setText(s['search_ell'])
        self._lbl_r1.setText(s['lbl_r1'])
        self._lbl_r2.setText(s['lbl_r2'])
        self._lbl_angle.setText(s['lbl_angle'])
        self._lbl_min_nb.setText(s['lbl_min_nb'])
        self._lbl_max_nb.setText(s['lbl_max_nb'])

        # Tab 2 — CRS
        self._grp_crs.setTitle(s['grp_crs'])
        self._lbl_crs_hint.setText(s['lbl_crs_hint'])

        # Tab 2 — Flow
        self._grp_flow.setTitle(s['grp_flow'])
        self.flow_vectors_check.setText(s['flow_check'])
        self._lbl_step_x.setText(s['lbl_step_x'])
        self._lbl_step_y.setText(s['lbl_step_y'])
        self.btn_refresh_flow.setText(s['btn_refresh_flow'])

        # Tab 4
        self.btn_crossval.setText(s['btn_crossval'])
        self.cv_table.setHorizontalHeaderLabels(s['cv_headers'])
        if self._loo_rmse is None:
            self.cv_stats_label.setText(s['cv_stats_init'])
