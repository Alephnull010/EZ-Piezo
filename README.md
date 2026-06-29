# EZ Piezo - QGIS Plugin

A QGIS plugin for generating **piezometric maps** (groundwater contour lines) through **Ordinary Kriging** interpolation, from point measurements on wells, boreholes or piezometers.

> Designed for hydrogeologists working with projected coordinate systems (default: Lambert 93 / EPSG:2154). Compatible with any projected CRS.

## Features

### Data input
- CSV import with automatic separator detection (`;` `,` `TAB` `|`)
- Manual entry and editing in the data table
- Duplicate coordinate detection (singular kriging matrix prevention)
- CRS validation: geographic CRS in degrees (e.g. EPSG:4326) are rejected; a projected CRS is required

### Variogram
- Omnidirectional or directional experimental variogram (azimuth + angular tolerance)
- 4 theoretical models: **spherical**, **exponential**, **gaussian**, **linear**
- Weighted least squares fitting (Cressie 1985 weighting: sigma proportional to 1/sqrt(N(h)))
- nugget <= sill enforced by reparametrization as (nugget, partial sill, range)
- Manual parameter override (nugget, sill, range) without automatic recomputation
- Plot of experimental variogram + fitted model with pair counts N(h) per lag

### Interpolation
- Block-vectorized Ordinary Kriging (4096 nodes/block): bounded memory regardless of grid resolution
- Configurable elliptic search neighborhood (major/minor radius, orientation, min/max neighbors) with batched solve per unique neighbor set
- Automatic jitter fallback when lu_solve returns non-finite values

### Output layers
| Layer | Format | Description |
|---|---|---|
| Kriging | GeoTIFF | Interpolated piezometric head, blue to red color ramp |
| Isopiezes | GPKG | Vector contour lines with major contours (thick, labeled) and minor contours (thin, unlabeled) |
| Ouvrages | Memory layer | Input points with ouvrage/x/y/z_ngf attributes |
| Flux | Memory layer | Groundwater flow vectors (-grad Z), shown as arrows |

- Optional NoData mask outside the convex hull of input points
- Automatic contour interval (round number rule) or manual
- Major contours every N lines (configurable 2-20, default 5), with offset control to align on preferred values
- `gdal.ContourGenerate` called with `useNoData=1, noDataValue=-9999`: no spurious contours at borders

### Cross-validation (LOO)
- Leave-One-Out using the same search neighborhood as the final map
- Variogram calibration statistics:
  - **Mean error** (target near 0) - unbiasedness
  - **RMSE** - absolute error
  - **Mean standardized error** = (measured - estimated) / sigma_k (target near 0)
  - **RMSSE** = RMS of standardized errors (target near 1; below 1 = overestimated uncertainty, above 1 = underestimated)
- Per-well table with standardized error column (red if |std. error| > 2)
- Measured vs estimated scatter plot with best-fit line
- Robust handling of points unresolved by the search ellipse (excluded from stats, shown as "-" in table)
- Runs automatically after each kriging computation

## Installation

### Requirements

**QGIS 3.16+** - numpy, scipy, matplotlib and GDAL are bundled with QGIS; no pip install required.

### From ZIP

**Plugins -> Manage and Install Plugins -> Install from ZIP** and select the archive.

### Manual copy

Copy the plugin folder to:

```
Windows : C:\Users\<user>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\piezo_kriging\
Linux   : ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/piezo_kriging/
macOS   : ~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/piezo_kriging/
```

Then: **Plugins -> Manage Plugins -> Installed -> check EZ Piezo**.

### Reload without restarting QGIS

After editing Python files: use the reload icon in the plugin manager, or install the *Plugin Reloader* plugin.

## Quick start

### Test dataset

The file `exemple_donnees.csv` (12 Lambert 93 points, NGF elevations) lets you test the plugin immediately.

### Workflow

1. **Data tab** - Load the CSV (separator `;`) or enter data manually
2. **Parameters tab** - Choose the variogram model, grid resolution (100 px is a good default), contour interval, and EPSG code (projected CRS required)
3. **Run Kriging** - 3 layers appear in QGIS; cross-validation runs automatically in tab 4
4. **Cross-validation tab** - Check that RMSSE is near 1; adjust variogram parameters if needed

### CSV format

```csv
Ouvrage;X;Y;Z
PZ01;843250.5;6518320.1;45.32
PZ02;843480.2;6518150.7;43.18
```

| Column | Description |
|---|---|
| Ouvrage | Well identifier (free text) |
| X, Y | Coordinates in a **projected** system (Lambert 93, UTM, etc.) |
| Z | Piezometric head in meters (NGF or any other vertical reference) |

Accepted separators: `;` `,` `TAB` `|` - auto-detected on import.

## Technical details

### Kriging engine (`kriging_engine.py`)

Pure Python module (no QGIS dependency), importable and testable standalone.

Pipeline:
```
compute_experimental_variogram
    -> fit_variogram (curve_fit, Cressie weighting)
    -> ordinary_kriging (Lagrange system n+1, scipy.linalg.lu_factor/lu_solve)
```

**Variogram fitting - Cressie (1985) weighting**
```python
sigma = 1 / sqrt(N(h))   # weight proportional to pair count per lag
```
Lags with more pairs carry more weight; extreme lags (few pairs) are naturally down-weighted.

**Reparametrization (nugget, psill, range)**

Fitting is done on the partial sill `psill = sill - nugget >= 0`, which guarantees `sill >= nugget` by construction and prevents non-physical decreasing variograms.

**Available models**

| Model | gamma(h) |
|---|---|
| Spherical | C0 + C * [1.5(h/a) - 0.5(h/a)^3] for h <= a, else C0+C |
| Exponential | C0 + C * [1 - exp(-3h/a)] |
| Gaussian | C0 + C * [1 - exp(-3(h/a)^2)] |
| Linear | C0 + slope * h |

C0 = nugget, C = partial sill, a = range.

**Performance**

- Global branch: K factorized once (`lu_factor`), grid processed in 4096-node blocks - peak memory is O(n x 4096), independent of total resolution
- Ellipse branch: ellipse membership vectorized per block (blk x n); nodes sharing the same neighbor set are grouped - one K factorization per unique group, batched RHS

### File structure

```
piezo_kriging/
+-- kriging_engine.py   # Pure geostatistical engine (numpy/scipy), no QGIS dependency
+-- piezo_dialog.py     # Qt5 multi-tab dialog (data, parameters, variogram, LOO)
+-- piezo_kriging.py    # QGIS integration: layer creation, styling, orchestration
+-- exemple_donnees.csv # Test dataset (12 Lambert 93 points)
+-- icons/icon.png
+-- metadata.txt
```

### Known limitations

- Synchronous execution: large grids (above 500x500) will block the QGIS UI thread; split into sub-areas if needed
- Ordinary Kriging only (no universal kriging or co-kriging)
- Stationary model assumed (no drift)
- Projected coordinate system required; decimal degrees (EPSG:4326) are detected and rejected

## License

GPL v2 - see LICENSE file.

Developed by Hugo LEBEL.
