"""
kriging_engine.py — Ordinary Kriging interpolation engine.

Implements:
  - Experimental variogram (omnidirectional or directional)
  - Variogram models: spherical, exponential, gaussian, linear
  - Ordinary Kriging (vectorized, with optional elliptic search neighborhood)
  - Leave-One-Out cross-validation
  - High-level run_kriging orchestrator

No dependency beyond numpy / scipy (both bundled with QGIS).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import curve_fit
from scipy.linalg import lu_factor, lu_solve

_KRIGING_BLOCK = 4096  # grid nodes per block (bounds peak memory for both branches)


# ──────────────────────────────────────────────
# Variogram models
# ──────────────────────────────────────────────

def spherical(h, nugget, sill, range_):
    h = np.asarray(h, dtype=float)
    gamma = np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        sill,
    )
    gamma[h == 0] = 0.0
    return gamma


def exponential(h, nugget, sill, range_):
    h = np.asarray(h, dtype=float)
    gamma = nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / range_))
    gamma[h == 0] = 0.0
    return gamma


def gaussian(h, nugget, sill, range_):
    h = np.asarray(h, dtype=float)
    gamma = nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * (h / range_) ** 2))
    gamma[h == 0] = 0.0
    return gamma


def linear(h, nugget, slope):
    """Linear variogram: γ(h) = nugget + slope × h  (no sill, no range)."""
    h = np.asarray(h, dtype=float)
    gamma = nugget + slope * h
    gamma[h == 0] = 0.0
    return gamma


VARIOGRAM_MODELS = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
    "linear": linear,
}


# ──────────────────────────────────────────────
# Experimental variogram
# ──────────────────────────────────────────────

def compute_experimental_variogram(
        coords, values, n_lags=15, max_lag=None,
        lag_size=None, min_pairs=0,
        direction=None, tolerance=90.0):
    """
    Compute binned experimental semi-variogram, optionally directional.

    Parameters
    ----------
    coords     : (N, 2) — X, Y positions
    values     : (N,)  — measured values
    n_lags     : int   — number of lag bins
    max_lag    : float — max lag distance (auto = 50 % of max dist); ignored when lag_size is set
    lag_size   : float — explicit bin width; when set, max range = lag_size × n_lags (Surfer convention)
    min_pairs  : int   — discard lags with fewer than this many pairs (0 = keep all non-empty)
    direction  : float — azimuth in degrees (0 = East); None = omnidirectional
    tolerance  : float — half-angle tolerance in degrees (90 = omnidirectional)

    Returns
    -------
    lag_centers, gamma, n_pairs
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)

    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = coords[idx_j, 0] - coords[idx_i, 0]
    dy = coords[idx_j, 1] - coords[idx_i, 1]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    sq_diffs = (values[idx_j] - values[idx_i]) ** 2

    # Directional filter
    if direction is not None and tolerance < 90.0:
        # Angles in [0, 180) — variogram is symmetric
        angles = np.degrees(np.arctan2(dy, dx)) % 180.0
        dir_norm = direction % 180.0
        ang_diff = np.abs(angles - dir_norm)
        ang_diff = np.minimum(ang_diff, 180.0 - ang_diff)
        mask_dir = ang_diff <= tolerance
        dists = dists[mask_dir]
        sq_diffs = sq_diffs[mask_dir]

    if len(dists) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    if lag_size is not None and lag_size > 0:
        # Surfer convention: range is fully determined by step × count
        max_lag_eff = lag_size * n_lags
        bins = np.arange(0.0, max_lag_eff + lag_size, lag_size)
    else:
        if max_lag is None:
            max_lag = 0.5 * dists.max()
        bins = np.linspace(0.0, max_lag, n_lags + 1)

    n_bins = len(bins) - 1
    lag_centers = 0.5 * (bins[:-1] + bins[1:])
    gamma = np.zeros(n_bins)
    n_pairs = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        in_bin = (dists >= bins[k]) & (dists < bins[k + 1])
        cnt = int(in_bin.sum())
        if cnt > 0:
            gamma[k] = 0.5 * sq_diffs[in_bin].mean()
            n_pairs[k] = cnt

    valid = n_pairs >= max(min_pairs, 1)
    return lag_centers[valid], gamma[valid], n_pairs[valid]


# ──────────────────────────────────────────────
# Variogram fitting
# ──────────────────────────────────────────────

def fit_variogram(lag_centers, gamma, model_name="spherical", n_pairs=None,
                  force_nugget_zero=False):
    """
    Fit a theoretical variogram to the experimental values.

    Returns
    -------
    params : dict
        Always contains 'nugget'. For 3-param models: also 'sill', 'range', slope=None.
        For 'linear': also 'slope', sill=None, range=None.
    model_func : callable(h) → γ(h)
    """
    if len(lag_centers) == 0:
        # Fallback with trivial params
        if model_name == "linear":
            params = {"nugget": 0.0, "sill": None, "range": None, "slope": 1e-6}
            return params, lambda h: linear(np.asarray(h), 0.0, 1e-6)
        else:
            params = {"nugget": 0.0, "sill": 1.0, "range": 1.0, "slope": None}
            return params, lambda h: VARIOGRAM_MODELS[model_name](np.asarray(h), 0.0, 1.0, 1.0)

    sill_est = float(gamma.max()) if gamma.max() > 0 else 1.0
    nugget_est = float(gamma.min()) if gamma.min() >= 0 else 0.0

    # Cressie (1985) weighted LS: weight ∝ N(h), i.e. sigma ∝ 1/sqrt(N(h))
    if n_pairs is not None and len(n_pairs) == len(lag_centers):
        sigma = 1.0 / np.sqrt(np.maximum(n_pairs, 1).astype(float))
    else:
        sigma = None

    if model_name == "linear":
        h_range = lag_centers.max() - lag_centers.min()
        slope_est = (gamma.max() - gamma.min()) / (h_range if h_range > 0 else 1.0)
        slope_bound_hi = sill_est * 10 / (lag_centers.max() or 1.0)

        if force_nugget_zero:
            def _linear_zero_nugget(h, slope):
                return linear(np.asarray(h), 0.0, slope)
            try:
                popt, _ = curve_fit(
                    _linear_zero_nugget, lag_centers, gamma,
                    p0=[max(slope_est, 1e-10)],
                    bounds=([1e-10], [slope_bound_hi]),
                    sigma=sigma, absolute_sigma=False, maxfev=5000,
                )
            except RuntimeError:
                popt = [max(slope_est, 1e-10)]
            nugget, slope = 0.0, float(popt[0])
        else:
            try:
                popt, _ = curve_fit(
                    linear, lag_centers, gamma,
                    p0=[nugget_est, max(slope_est, 1e-10)],
                    bounds=([0.0, 1e-10], [sill_est * 3, slope_bound_hi]),
                    sigma=sigma, absolute_sigma=False, maxfev=5000,
                )
            except RuntimeError:
                popt = [nugget_est, max(slope_est, 1e-10)]
            nugget, slope = float(popt[0]), float(popt[1])

        params = {"nugget": nugget, "sill": None, "range": None, "slope": slope}

        def fitted(h):
            return linear(np.asarray(h, dtype=float), nugget, slope)

        return params, fitted

    # 3-parameter models — fit in (nugget, psill, range_) to enforce sill = nugget + psill ≥ nugget
    model_func = VARIOGRAM_MODELS[model_name]
    range_est = float(lag_centers[len(lag_centers) // 2]) if len(lag_centers) > 0 else 1.0
    psill_est = max(sill_est - nugget_est, 1e-10)

    def _model_psill(h, nugget, psill, range_):
        return model_func(h, nugget, nugget + psill, range_)

    p0 = [nugget_est, psill_est, range_est]
    bounds = (
        [0.0, 1e-10, 1e-6],
        [sill_est * 2, sill_est * 5, lag_centers.max() * 2],
    )
    try:
        popt, _ = curve_fit(_model_psill, lag_centers, gamma, p0=p0, bounds=bounds,
                            sigma=sigma, absolute_sigma=False, maxfev=5000)
    except RuntimeError:
        popt = p0

    nugget, psill, range_ = float(popt[0]), float(popt[1]), float(popt[2])
    sill = nugget + psill
    params = {"nugget": nugget, "sill": sill, "range": range_, "slope": None}

    def fitted(h):
        return model_func(np.asarray(h, dtype=float), nugget, sill, range_)

    return params, fitted


# ──────────────────────────────────────────────
# Ordinary Kriging
# ──────────────────────────────────────────────

def find_duplicate_coords(coords, tol=1e-3):
    """Return list of (i, j) index pairs where distance between coords[i] and coords[j] < tol."""
    coords = np.asarray(coords, dtype=float)
    dists = squareform(pdist(coords))
    idx_i, idx_j = np.triu_indices(len(coords), k=1)
    close = dists[idx_i, idx_j] < tol
    return list(zip(idx_i[close].tolist(), idx_j[close].tolist()))


def ordinary_kriging(
        coords, values, grid_x, grid_y, variogram_func,
        search_params=None):
    """
    Ordinary Kriging on a regular grid.

    Parameters
    ----------
    coords, values  : known data
    grid_x, grid_y  : 1-D grid node arrays
    variogram_func  : callable(h) → γ(h)
    search_params   : None → use all data (vectorised).
                      dict → elliptic search neighborhood:
                        use_all      : bool (True = ignore search params)
                        radius1      : float (major semi-axis)
                        radius2      : float (minor semi-axis)
                        angle        : float (degrees, rotation of major axis from East)
                        min_neighbors: int
                        max_neighbors: int

    Returns
    -------
    Z        : (ny, nx) — kriging estimates (NaN where no neighbor found)
    variance : (ny, nx) — kriging variance
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)
    ny_g, nx_g = len(grid_y), len(grid_x)

    gx_m, gy_m = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx_m.ravel(), gy_m.ravel()])
    n_grid = grid_pts.shape[0]

    use_search = (
        search_params is not None
        and not search_params.get("use_all", True)
    )

    if not use_search:
        # ── Block-vectorised: factorize K once, process grid in chunks ──
        dist_matrix = squareform(pdist(coords))
        gamma_matrix = variogram_func(dist_matrix)

        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = gamma_matrix
        K[:n, n] = 1.0
        K[n, :n] = 1.0

        try:
            lu = lu_factor(K)
        except Exception:
            K[:n, :n] += np.eye(n) * 1e-6
            lu = lu_factor(K)

        Z_flat = np.empty(n_grid)
        var_flat = np.empty(n_grid)
        _jittered = False

        for blk_s in range(0, n_grid, _KRIGING_BLOCK):
            blk_e = min(blk_s + _KRIGING_BLOCK, n_grid)
            dist_blk = cdist(grid_pts[blk_s:blk_e], coords)    # (blk, n)
            gamma_blk = variogram_func(dist_blk)                # (blk, n)
            rhs = np.ones((n + 1, blk_e - blk_s))
            rhs[:n, :] = gamma_blk.T
            w = lu_solve(lu, rhs)                               # (n+1, blk)
            if not np.isfinite(w).all() and not _jittered:
                K[:n, :n] += np.eye(n) * 1e-6
                lu = lu_factor(K)
                w = lu_solve(lu, rhs)
                _jittered = True
            Z_flat[blk_s:blk_e] = w[:n, :].T @ values
            var_flat[blk_s:blk_e] = (
                np.sum(w[:n, :].T * gamma_blk, axis=1) + w[n, :]
            )

        Z = Z_flat.reshape(ny_g, nx_g)
        variance = var_flat.reshape(ny_g, nx_g)

    else:
        # ── Elliptic search: vectorised membership + batched solve per unique neighbourhood ──
        radius1 = float(search_params.get("radius1", np.inf))
        radius2 = float(search_params.get("radius2", radius1))
        angle_deg = float(search_params.get("angle", 0.0))
        min_nb = int(search_params.get("min_neighbors", 1))
        max_nb = int(search_params.get("max_neighbors", n))

        cos_a = np.cos(np.radians(angle_deg))
        sin_a = np.sin(np.radians(angle_deg))

        Z_flat = np.full(n_grid, np.nan)
        var_flat = np.full(n_grid, np.nan)

        for blk_s in range(0, n_grid, _KRIGING_BLOCK):
            blk_e = min(blk_s + _KRIGING_BLOCK, n_grid)
            blk_pts = grid_pts[blk_s:blk_e]                    # (blk, 2)

            # Vectorised ellipse membership for the whole block at once
            dx = blk_pts[:, 0][:, None] - coords[:, 0][None, :]  # (blk, n)
            dy = blk_pts[:, 1][:, None] - coords[:, 1][None, :]
            dx_rot = dx * cos_a + dy * sin_a
            dy_rot = -dx * sin_a + dy * cos_a
            in_ell = (dx_rot / radius1) ** 2 + (dy_rot / radius2) ** 2 <= 1.0
            dist_sq = dx ** 2 + dy ** 2                         # (blk, n)

            # Group block nodes by their final (post-truncation) neighbour set
            groups = {}  # bytes_key -> (nb_idx, [global_idx…], [dist_gp…])
            for loc in range(blk_e - blk_s):
                nb = np.where(in_ell[loc])[0]
                if len(nb) < min_nb:
                    continue
                if len(nb) > max_nb:
                    nb = nb[np.argsort(dist_sq[loc, nb])[:max_nb]]
                    nb = np.sort(nb)
                key = nb.tobytes()
                d_gp = np.sqrt(dist_sq[loc, nb])
                if key not in groups:
                    groups[key] = (nb.copy(), [], [])
                groups[key][1].append(blk_s + loc)
                groups[key][2].append(d_gp)

            # One K factorization + batched RHS solve per unique neighbourhood
            for nb_idx, global_ids, dist_gp_list in groups.values():
                sub_c = coords[nb_idx]
                sub_v = values[nb_idx]
                n_s = len(sub_v)

                gamma_sub = variogram_func(squareform(pdist(sub_c)))
                K_s = np.zeros((n_s + 1, n_s + 1))
                K_s[:n_s, :n_s] = gamma_sub
                K_s[:n_s, n_s] = 1.0
                K_s[n_s, :n_s] = 1.0

                try:
                    lu_s = lu_factor(K_s)
                except Exception:
                    K_s[:n_s, :n_s] += np.eye(n_s) * 1e-6
                    try:
                        lu_s = lu_factor(K_s)
                    except Exception:
                        continue

                m = len(global_ids)
                gamma_gp = variogram_func(np.array(dist_gp_list))  # (m, n_s)
                rhs = np.ones((n_s + 1, m))
                rhs[:n_s, :] = gamma_gp.T
                w = lu_solve(lu_s, rhs)                         # (n_s+1, m)
                if not np.isfinite(w).all():
                    K_s[:n_s, :n_s] += np.eye(n_s) * 1e-6
                    lu_s = lu_factor(K_s)
                    w = lu_solve(lu_s, rhs)
                if not np.isfinite(w).all():
                    continue

                idx_arr = np.array(global_ids)
                Z_flat[idx_arr] = w[:n_s, :].T @ sub_v
                var_flat[idx_arr] = (
                    np.sum(w[:n_s, :].T * gamma_gp, axis=1) + w[n_s, :]
                )

        Z = Z_flat.reshape(ny_g, nx_g)
        variance = var_flat.reshape(ny_g, nx_g)

    return Z, variance


# ──────────────────────────────────────────────
# Leave-One-Out cross-validation
# ──────────────────────────────────────────────

def cross_validate_loo(coords, values, variogram_func, search_params=None):
    """
    Leave-One-Out cross-validation using Ordinary Kriging.

    Parameters
    ----------
    search_params : dict or None
        Passed through to ordinary_kriging — must match the params used for
        the final kriging run so the validation reflects the same configuration.

    Returns
    -------
    dict with: measured, estimated, errors, kvar,
               mean_error, rmse,
               mean_std_error (target ≈ 0), rmsse (target ≈ 1)
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)
    estimated = np.full(n, np.nan)
    kvar = np.full(n, np.nan)

    for i in range(n):
        train_c = np.delete(coords, i, axis=0)
        train_v = np.delete(values, i)
        Z, V = ordinary_kriging(
            train_c, train_v,
            [float(coords[i, 0])], [float(coords[i, 1])],
            variogram_func,
            search_params=search_params,
        )
        z_val = float(Z[0, 0])
        v_val = float(V[0, 0])
        if np.isfinite(z_val) and np.isfinite(v_val) and v_val > 0:
            estimated[i] = z_val
            kvar[i] = v_val

    errors = values - estimated
    valid = np.isfinite(estimated) & np.isfinite(kvar)
    ev = errors[valid]
    sev = ev / np.sqrt(kvar[valid])
    return {
        "measured": values,
        "estimated": estimated,
        "errors": errors,
        "kvar": kvar,
        "valid": valid,
        "n_failed": int((~valid).sum()),
        "mean_error": float(np.mean(ev)) if ev.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(ev**2))) if ev.size else float("nan"),
        "mean_std_error": float(np.mean(sev)) if sev.size else float("nan"),
        "rmsse": float(np.sqrt(np.mean(sev**2))) if sev.size else float("nan"),
    }


# ──────────────────────────────────────────────
# Flow vectors
# ──────────────────────────────────────────────

def compute_flow_vectors(Z, grid_x, grid_y, step_x=10, step_y=10):
    """
    Compute downsampled groundwater flow vectors from the kriged head grid.
    Flow direction = -∇Z (direction of decreasing hydraulic head).

    Parameters
    ----------
    Z        : (ny, nx) array — kriged head.
               Z[i, j] corresponds to (grid_x[j], grid_y[i]),
               with grid_y[0] = ymin (south) and grid_y[ny-1] = ymax (north).
    grid_x   : (nx,) 1-D array of X coordinates (increasing eastward)
    grid_y   : (ny,) 1-D array of Y coordinates (increasing northward)
    step_x   : sampling interval in grid nodes along X
    step_y   : sampling interval in grid nodes along Y

    Returns
    -------
    list of dicts {x, y, dx, dy, magnitude}
      dx, dy    : unit-normalised flow vector components
      magnitude : hydraulic gradient |∇Z| in m/m
    """
    Z = np.asarray(Z, dtype=float)
    ny, nx = Z.shape
    if ny < 2 or nx < 2:
        return []

    dy_sp = (grid_y[-1] - grid_y[0]) / max(ny - 1, 1)
    dx_sp = (grid_x[-1] - grid_x[0]) / max(nx - 1, 1)

    # Z[i] corresponds to grid_y[i] which increases northward (same direction as Y).
    # np.gradient(Z, dy_sp, dx_sp) gives dZ/d(northward) and dZ/d(eastward) directly.
    dZ_drow, dZ_dcol = np.gradient(Z, dy_sp, dx_sp)
    dZ_dy = dZ_drow    # rows increase northward → no sign correction needed
    dZ_dx = dZ_dcol

    # Flow = -∇Z (toward lower head)
    flow_x = -dZ_dx
    flow_y = -dZ_dy
    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)

    # Subsample on a regular node grid; centred offset avoids boundary artefacts
    row_idx = np.arange(step_y // 2, ny, step_y)
    col_idx = np.arange(step_x // 2, nx, step_x)

    vectors = []
    for ri in row_idx:
        for ci in col_idx:
            if np.isnan(Z[ri, ci]):
                continue
            mag = float(magnitude[ri, ci])
            if mag < 1e-10:
                continue
            vectors.append({
                "x": float(grid_x[ci]),
                "y": float(grid_y[ri]),   # Z[ri] ↔ grid_y[ri]
                "dx": float(flow_x[ri, ci] / mag),
                "dy": float(flow_y[ri, ci] / mag),
                "magnitude": mag,
            })
    return vectors


# ──────────────────────────────────────────────
# High-level helper
# ──────────────────────────────────────────────

def nice_contour_interval(z_range, target_count=10):
    """Return a 'round' contour interval for ~target_count contours over z_range."""
    if z_range <= 0:
        return 0.1
    rough = z_range / target_count
    magnitude = 10.0 ** np.floor(np.log10(rough))
    norm = rough / magnitude
    if norm < 1.5:
        nice = 1.0
    elif norm < 3.5:
        nice = 2.0
    elif norm < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def run_kriging(coords, values, nx=100, ny=100, n_lags=15, max_lag=None,
                lag_size=None, model_name="spherical", pad_percent=5,
                direction=None, tolerance=90.0, nodata_outside_hull=False,
                search_params=None):
    """
    End-to-end kriging pipeline.

    Parameters
    ----------
    coords, values   : input data
    nx, ny           : grid node counts on X and Y axes
    n_lags           : lag bin count (ignored if lag_size is set)
    max_lag          : max lag distance (auto if None)
    lag_size         : explicit lag bin width (overrides n_lags)
    model_name       : 'spherical' | 'exponential' | 'gaussian' | 'linear'
    pad_percent      : grid extent padding as % of data range
    direction        : variogram direction in degrees (None = omnidirectional)
    tolerance        : angular tolerance in degrees
    nodata_outside_hull : set Z=NaN outside the convex hull of input points
    search_params    : dict for elliptic search neighborhood (see ordinary_kriging)

    Returns
    -------
    dict: grid_x, grid_y, Z, variance, variogram_params, lag_centers,
          gamma_exp, n_pairs, vario_func
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    lag_centers, gamma_exp, n_pairs = compute_experimental_variogram(
        coords, values, n_lags=n_lags, max_lag=max_lag,
        lag_size=lag_size, direction=direction, tolerance=tolerance,
    )

    vario_params, vario_func = fit_variogram(lag_centers, gamma_exp, model_name, n_pairs)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    dx = (xmax - xmin) * pad_percent / 100.0
    dy = (ymax - ymin) * pad_percent / 100.0
    grid_x = np.linspace(xmin - dx, xmax + dx, nx)
    grid_y = np.linspace(ymin - dy, ymax + dy, ny)

    Z, variance = ordinary_kriging(
        coords, values, grid_x, grid_y, vario_func, search_params=search_params
    )

    if nodata_outside_hull and len(coords) >= 3:
        from scipy.spatial import Delaunay
        hull = Delaunay(coords)
        gx_m, gy_m = np.meshgrid(grid_x, grid_y)
        grid_flat = np.column_stack([gx_m.ravel(), gy_m.ravel()])
        outside = hull.find_simplex(grid_flat) < 0
        Z[outside.reshape(ny, nx)] = np.nan

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "Z": Z,
        "variance": variance,
        "variogram_params": vario_params,
        "lag_centers": lag_centers,
        "gamma_exp": gamma_exp,
        "n_pairs": n_pairs,
        "vario_func": vario_func,
    }
