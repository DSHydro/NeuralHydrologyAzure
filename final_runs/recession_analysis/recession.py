"""
Baseflow Recession Constant calculation.

Translated from MATLAB (Copyright C 2020, GNU Public License Version 3).

References:
    Safeeq et al. (2013), Hydrological Processes, 27(5), pp.655-668.
    Posavec et al. (2006), Groundwater, 44(5), pp.764-767.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


# ---------------------------------------------------------------------------
# Lyne-Hollick baseflow filter
# ---------------------------------------------------------------------------

def util_LyneHollickFilter(Q, filter_parameter=0.925, nr_passes=1,
                           threshold_type='pass'):
    """
    Estimate baseflow using the Lyne-Hollick recursive digital filter.

    Translated directly from the MATLAB util_LyneHollickFilter.

    Parameters
    ----------
    Q : array-like
        Streamflow [mm/timestep].
    filter_parameter : float
        Filter parameter, between 0 and 1 (default 0.925).
    nr_passes : int
        Number of filter passes (default 1, forward pass only).
    threshold_type : str
        How to threshold the quickflow component:
        - 'pass'      : set negative values to 0 after each pass (default)
        - 'timestep'  : set negative values to 0 after each timestep
        - 'end'       : set negative values to 0 only at the very end
        - 'none'      : no thresholding (baseflow may exceed streamflow)

    Returns
    -------
    Q_b : np.ndarray
        Baseflow [mm/timestep].

    Notes
    -----
    NaN values are temporarily replaced with median(Q) before filtering,
    then restored. This matches the MATLAB behaviour.
    Passes alternate direction: pass 1 = forward, pass 2 = backward
    (flip), pass 3 = forward, etc. — matching MATLAB's flip(Q_b) on
    each subsequent pass.
    Initial condition follows Su et al. (2016): Q_f[0] = Q[0] - min(Q).
    """
    if not (0 < filter_parameter <= 1):
        raise ValueError('filter_parameter must be between 0 and 1.')
    if nr_passes < 1 or nr_passes != int(nr_passes):
        raise ValueError('nr_passes must be a positive integer.')

    Q = np.asarray(Q, dtype=float)

    # --- NaN handling: replace with median, restore after ---
    # Matches MATLAB: Q_tmp(isnan(Q)) = median(Q, 'omitnan')
    nan_mask = np.isnan(Q)
    Q_tmp = Q.copy()
    Q_tmp[nan_mask] = np.nanmedian(Q)

    def _lyne_hollick_pass(q, threshold_type):
        """Single forward pass of the filter (always on the input as-is)."""
        # Matches MATLAB's inner LyneHollickFilter helper function
        n = len(q)
        a = filter_parameter
        Q_f = np.full(n, np.nan)

        # Initial condition from Su et al. (2016)
        # MATLAB: Q_f(1) = Q(1) - min(Q)
        Q_f[0] = q[0] - np.min(q)

        if threshold_type == 'timestep':
            # MATLAB: constrain Q_f after each timestep
            for i in range(1, n):
                Q_f[i] = a * Q_f[i-1] + ((1 + a) / 2) * (q[i] - q[i-1])
                if Q_f[i] < 0:
                    Q_f[i] = 0.0
        else:
            # MATLAB: no per-timestep clipping
            for i in range(1, n):
                Q_f[i] = a * Q_f[i-1] + ((1 + a) / 2) * (q[i] - q[i-1])

        if threshold_type == 'pass':
            # MATLAB: Q_f(Q_f<0) = 0  after each pass
            Q_f[Q_f < 0] = 0.0

        # baseflow = streamflow - quickflow
        Q_b = q - Q_f
        return Q_b

    # --- First pass (always forward) ---
    # MATLAB: Q_b = LyneHollickFilter(Q_tmp, filter_parameter, threshold_type)
    Q_b = _lyne_hollick_pass(Q_tmp, threshold_type)

    # --- Subsequent passes on flipped array ---
    # MATLAB: for nr = 2:nr_passes
    #             Q_b = LyneHollickFilter(flip(Q_b), ...)
    #         end
    for _ in range(2, nr_passes + 1):
        Q_b = _lyne_hollick_pass(Q_b[::-1], threshold_type)

    # --- Restore NaNs ---
    # MATLAB: Q_b(isnan(Q)) = NaN
    Q_b[nan_mask] = np.nan

    # --- Cap baseflow at streamflow (unless threshold_type='none') ---
    # MATLAB: if strcmp(threshold_type,'none') / else / Q_b(Q_b>Q)=Q(Q_b>Q)
    if threshold_type != 'none':
        Q_b = np.where(Q_b > Q, Q, Q_b)

    return Q_b


# ---------------------------------------------------------------------------
# Data check
# ---------------------------------------------------------------------------

def util_DataCheck(Q, t, P=None, PET=None, T=None):
    """
    Check input data for common issues before signature calculation.

    Translated directly from MATLAB util_DataCheck.

    Parameters
    ----------
    Q : array-like
        Streamflow [mm/timestep].
    t : array-like
        Time array (datetime64 or numeric).
    P : array-like, optional
        Precipitation [mm/timestep].
    PET : array-like, optional
        Potential evapotranspiration [mm/timestep].
    T : array-like, optional
        Temperature [degC].

    Returns
    -------
    error_flag : int
        0 = no error, 1 = warning, 2 = error in data check.
    error_str : str
        Description of any error/warning.
    timestep : np.timedelta64 or float
        Median timestep of the time series.
    t : np.ndarray
        Time array as datetime64 (converted if numeric input).
    """
    error_flag = 0
    error_str = ''

    Q = np.asarray(Q, dtype=float)
    t = np.asarray(t)

    # --- Timestep checks ---
    # MATLAB: if isnumeric(t) -> convert from datenum to datetime
    # Python: if t is numeric -> convert to datetime64
    # MATLAB datenums are days since 0-Jan-0000; datetime64 epoch is 1970-01-01
    # The offset between the two epochs is 719529 days
    if np.issubdtype(t.dtype, np.floating) or np.issubdtype(t.dtype, np.integer):
        error_flag = 1
        error_str = 'Warning: Converted numeric t to datetime64. ' + error_str
        t = (t - 719529).astype('datetime64[D]')

    # MATLAB: timesteps = diff(t); timestep = median(timesteps)
    timesteps = np.diff(t)
    timestep = np.median(timesteps)

    # MATLAB: if any(diff(timesteps) ~= 0)
    if len(np.unique(timesteps)) > 1:
        error_flag = max(error_flag, 1)
        error_str = ('Warning: Record is not continuous '
                     '(some timesteps are missing). ' + error_str)

    # --- Q checks ---
    # MATLAB: if min(Q) < 0
    if np.nanmin(Q) < 0:
        error_flag = 2
        error_str = 'Error: Negative values in flow series. ' + error_str
        return error_flag, error_str, timestep, t

    # MATLAB: if all(Q == 0)
    if np.all(Q == 0):
        error_flag = 2
        error_str = 'Error: Only zero flow in flow series. ' + error_str
        return error_flag, error_str, timestep, t

    # MATLAB: if length(Q) ~= length(t)
    if len(Q) != len(t):
        error_flag = 2
        error_str = ('Error: Flow series and time vector have different lengths. '
                     + error_str)
        return error_flag, error_str, timestep, t

    # MATLAB: if any(isnan(Q))
    if np.any(np.isnan(Q)):
        error_flag = max(error_flag, 1)
        error_str = 'Warning: Ignoring NaNs in streamflow data. ' + error_str

    # MATLAB: if all(isnan(Q))
    if np.all(np.isnan(Q)):
        error_flag = 2
        error_str = 'Error: Only NaNs in streamflow data. ' + error_str
        return error_flag, error_str, timestep, t

    # MATLAB: if length(Q) < 30
    if len(Q) < 30:
        error_flag = max(error_flag, 1)
        error_str = 'Warning: Extremely short time series. ' + error_str

    # --- Optional P checks ---
    # MATLAB: if ~isempty(P)
    if P is not None:
        P = np.asarray(P, dtype=float)

        if np.any(np.isnan(P)):
            error_flag = max(error_flag, 1)
            error_str = 'Warning: Ignoring NaNs in precipitation data. ' + error_str

        if np.all(np.isnan(P)):
            error_flag = 2
            error_str = 'Error: Only NaNs in precipitation data. ' + error_str
            return error_flag, error_str, timestep, t

        if len(Q) != len(P):
            error_flag = 2
            error_str = ('Error: Precipitation and flow series have different lengths. '
                         + error_str)
            return error_flag, error_str, timestep, t

        if np.nanmin(P) < 0:
            error_flag = 2
            error_str = 'Error: Negative values in precipitation series. ' + error_str
            return error_flag, error_str, timestep, t

    # --- Optional PET checks ---
    # MATLAB: if ~isempty(PET)
    if PET is not None:
        PET = np.asarray(PET, dtype=float)

        if np.any(np.isnan(PET)):
            error_flag = max(error_flag, 1)
            error_str = ('Warning: Ignoring NaNs in potential evapotranspiration data. '
                         + error_str)

        if np.all(np.isnan(PET)):
            error_flag = 2
            error_str = ('Error: Only NaNs in potential evapotranspiration data. '
                         + error_str)
            return error_flag, error_str, timestep, t

        if len(Q) != len(PET):
            error_flag = 2
            error_str = ('Error: Potential evapotranspiration and flow series have '
                         'different lengths. ' + error_str)
            return error_flag, error_str, timestep, t

        # Note: MATLAB raises a warning (flag=1) for negative PET, not an error
        if np.nanmin(PET) < 0:
            error_flag = max(error_flag, 1)
            error_str = ('Warning: Negative values in potential evapotranspiration series. '
                         + error_str)

    # --- Optional T checks ---
    # MATLAB: if ~isempty(T)
    if T is not None:
        T = np.asarray(T, dtype=float)

        if np.any(np.isnan(T)):
            error_flag = max(error_flag, 1)
            error_str = 'Warning: Ignoring NaNs in temperature data. ' + error_str

        if np.all(np.isnan(T)):
            error_flag = 2
            error_str = 'Error: Only NaNs in temperature data. ' + error_str
            return error_flag, error_str, timestep, t

        if len(Q) != len(T):
            error_flag = 2
            error_str = ('Error: Temperature and flow series have different lengths. '
                         + error_str)
            return error_flag, error_str, timestep, t

        # MATLAB: if min(T) < -273.15
        if np.nanmin(T) < -273.15:
            error_flag = 2
            error_str = ('Error: Temperature cannot be less than -273.15 degC. '
                         + error_str)
            return error_flag, error_str, timestep, t

    return error_flag, error_str, timestep, t


# ---------------------------------------------------------------------------
# Recession segment identification
# ---------------------------------------------------------------------------

def util_RecessionSegments(Q, t,
                           recession_length=5,
                           n_start=1,
                           eps=0.0,
                           start_of_recession='peak',
                           filter_par=0.925):
    """
    Identify individual recession segments in a streamflow time series.

    Parameters
    ----------
    Q : array-like
        Streamflow [mm/timestep].
    t : array-like
        Time array (datetime64 or numeric, uniform spacing assumed).
    recession_length : int
        Minimum recession length in days (default 5).
    n_start : int
        Days to remove at beginning of each recession (default 1).
    eps : float
        Allowed increase in flow during recession (default 0).
    start_of_recession : str
        'peak' or 'baseflow'.
    filter_par : float
        Lyne-Hollick filter parameter (used when start_of_recession='baseflow').

    Returns
    -------
    flow_section : np.ndarray, shape (n_segments, 2)
        Start and end indices of each recession segment.
    error_flag : int
        0 = no error, 1 = warning, 3 = error.
    error_str : str
        Description of any error/warning.
    """
    Q = np.asarray(Q, dtype=float)
    t = np.asarray(t)
    error_flag = 0
    error_str = ''

    # Warn if eps is large relative to median flow
    median_Q = np.nanmedian(Q)
    if eps > median_Q / 100:
        error_flag = 1
        error_str = ('Warning: eps set to a value larger than 1% of median(Q). '
                     'High eps values can lead to problematic recession selection. ' + error_str)

    # Zero-flow days are treated as NaN
    iszero = (Q == 0)
    Q = Q.copy()
    Q[iszero] = np.nan

    # Infer timestep length in days
    if np.issubdtype(t.dtype, np.datetime64):
        dt_days = (t[1] - t[0]) / np.timedelta64(1, 'D')
    else:
        dt_days = float(t[1] - t[0])

    len_decrease = recession_length / dt_days

    # Find timesteps with decreasing (or eps-allowed) flow
    decreasing_flow = Q[1:] < (Q[:-1] + eps)

    # Start on a non-decreasing point
    start_point_arr = np.where(~decreasing_flow)[0]
    if len(start_point_arr) == 0:
        error_flag = 3
        error_str = 'Error: No non-decreasing points found. ' + error_str
        return np.empty((0, 2), dtype=int), error_flag, error_str

    start_point = start_point_arr[0]
    decreasing_flow = decreasing_flow[start_point:]

    # Find transitions between decreasing / non-decreasing
    changes = np.where(np.diff(decreasing_flow.astype(int)) != 0)[0]

    # Pair up starts and ends
    n_pairs = 2 * (len(changes) // 2)
    changes = changes[:n_pairs]
    flow_change = changes.reshape(-1, 2)  # (start, end) of each decreasing section

    # Keep only sections long enough
    long_enough = (flow_change[:, 1] - flow_change[:, 0]) >= (len_decrease + n_start)
    flow_section = flow_change[long_enough].copy()
    flow_section += start_point          # shift back to original indices
    flow_section[:, 0] += n_start        # remove n_start days from beginning

    # Restore zeros
    Q[iszero] = 0.0

    # --- Apply start_of_recession logic ---
    if start_of_recession == 'peak':
        if len(flow_section) == 0:
            error_flag = 3
            error_str = ('Error: No long enough recession periods, '
                         'consider setting eps > 0. ' + error_str)
            return flow_section, error_flag, error_str
        if len(flow_section) < 10:
            error_flag = max(error_flag, 1)
            error_str = ('Warning: Fewer than 10 recession segments extracted, '
                         'results might not be robust. ' + error_str)

    elif start_of_recession == 'baseflow':
        Q_b = util_LyneHollickFilter(Q, filter_parameter=filter_par, nr_passes=1)
        isbaseflow = (Q_b == Q)

        keep = []
        for i in range(len(flow_section)):
            seg = isbaseflow[flow_section[i, 0]:flow_section[i, 1] + 1]
            if not np.any(seg):
                continue  # no baseflow point — drop this segment
            isb_start = np.where(seg)[0][0]
            flow_section[i, 0] += isb_start
            if flow_section[i, 1] >= flow_section[i, 0] + 3:
                keep.append(i)

        flow_section = flow_section[keep]

        if len(flow_section) == 0:
            error_flag = 3
            error_str = ('Error: No long enough baseflow recession periods, '
                         'consider increasing filter_par. ' + error_str)
            return flow_section, error_flag, error_str
        if len(flow_section) < 10:
            error_flag = max(error_flag, 1)
            error_str = ('Warning: Fewer than 10 recession segments extracted, '
                         'results might not be robust. ' + error_str)
    else:
        raise ValueError(f"start_of_recession must be 'peak' or 'baseflow', got '{start_of_recession}'")

    return flow_section, error_flag, error_str


# ---------------------------------------------------------------------------
# Master Recession Curve (nonparametric_analytic only)
# ---------------------------------------------------------------------------

# def util_MasterRecessionCurve(Q, flow_section,
#                                fit_method='nonparametric_analytic',
#                                match_method='log'):
#     """
#     Fit a Master Recession Curve to recession segments.

#     Parameters
#     ----------
#     Q : array-like
#         Streamflow [mm/timestep].
#     flow_section : np.ndarray, shape (n, 2)
#         Start/end indices for each recession segment.
#     fit_method : str
#         Only 'nonparametric_analytic' is implemented.
#     match_method : str
#         'linear' or 'log' spacing of interpolation points.

#     Returns
#     -------
#     MRC : np.ndarray, shape (m, 2)
#         Columns are [relative time, flow].
#     """
#     Q = np.asarray(Q, dtype=float)
#     flow_section = np.asarray(flow_section, dtype=int)

#     if fit_method != 'nonparametric_analytic':
#         raise NotImplementedError(f"fit_method '{fit_method}' is not implemented. "
#                                   "Only 'nonparametric_analytic' is supported.")

#     jitter_size = 1e-8
#     numflows = 500
#     rng = np.random.default_rng(0)  # fixed seed for reproducibility

#     numsegments = len(flow_section)

#     # Sort segments by initial flow value (descending)
#     flow_init_value = Q[flow_section[:, 0]]
#     sortind = np.argsort(flow_init_value)[::-1]
#     running_min = float(np.max(flow_init_value))

#     # Build segments list with jitter
#     segments = []
#     for i in range(numsegments):
#         seg = Q[flow_section[sortind[i], 0]: flow_section[sortind[i], 1] + 1].copy()
#         seg[1:] += rng.normal(0, jitter_size, len(seg) - 1)
#         seg = np.abs(seg) + 1e-20
#         seg = np.sort(seg)[::-1]  # ensure descending
#         segments.append(seg)

#     # Flow value grid for matching
#     max_flow = max(s.max() for s in segments)
#     min_flow = min(s.min() for s in segments)
#     if min_flow <= 0:
#         min_flow = jitter_size

#     if match_method == 'linear':
#         flow_vals = np.linspace(max_flow, min_flow, numflows)
#     elif match_method == 'log':
#         frac_log = 0.2
#         gridspace = (max_flow - min_flow) / numflows
#         linear_part = np.linspace(max_flow - gridspace / 2,
#                                    min_flow + gridspace / 2,
#                                    numflows - int(frac_log * numflows))
#         log_part = np.logspace(np.log10(max_flow), np.log10(min_flow),
#                                 int(frac_log * numflows))
#         flow_vals = np.sort(np.unique(np.concatenate([linear_part, log_part])))[::-1]
#         flow_vals[-1] = min_flow
#         flow_vals[0] = max_flow
#         numflows = len(flow_vals)
#     else:
#         raise ValueError(f"match_method must be 'linear' or 'log', got '{match_method}'")

#     # First pass: remove segments with no interpolated values
#     short_segs = []
#     for i, seg in enumerate(segments):
#         fmax_index = np.searchsorted(-flow_vals, -seg[0], side='left')
#         if seg[-1] <= flow_vals[-1]:
#             fmin_index = numflows - 1
#         else:
#             fmin_index = np.searchsorted(-flow_vals, -seg[-1], side='left') - 1
#         nf = fmin_index - fmax_index + 1
#         if nf <= 1:
#             short_segs.append(i)

#     # Remove short segments
#     for i in sorted(short_segs, reverse=True):
#         segments.pop(i)

#     if len(segments) == 0:
#         return np.array([[np.nan, np.nan]])

#     numsegments = len(segments)
#     running_min = max(s[0] for s in segments)

#     # Recompute flow_vals after removal
#     max_flow = max(s.max() for s in segments)
#     min_flow = min(s.min() for s in segments)
#     flow_vals = flow_vals[(flow_vals <= max_flow) & (flow_vals >= min_flow)]
#     numflows = len(flow_vals)

#     # Build sparse least-squares system
#     # Unknowns: [lag_2, ..., lag_N,  mrc_time_1, ..., mrc_time_M]
#     #   total unknowns = (numsegments - 1) + numflows
#     n_unknowns = (numsegments - 1) + numflows
#     rows_list = []
#     cols_list = []
#     vals_list = []
#     b_list = []
#     row = 0
#     bad_segs = []

#     for i, seg in enumerate(segments):
#         # Extend segment upward to running_min if there is a gap
#         if seg[0] < running_min:
#             seg = np.concatenate([[running_min], seg])

#         fmax_index = np.searchsorted(-flow_vals, -seg[0], side='left')
#         if seg[-1] <= flow_vals[-1]:
#             fmin_index = numflows - 1
#         else:
#             fmin_index = np.searchsorted(-flow_vals, -seg[-1], side='left') - 1

#         nf = fmin_index - fmax_index + 1
#         if nf == 0:
#             bad_segs.append(i)
#             continue

#         # Interpolate segment onto flow_vals
#         # seg is descending flow vs implicit index 0..len-1
#         x_seg = np.arange(len(seg), dtype=float)
#         interp_vals = np.interp(flow_vals[fmax_index: fmin_index + 1], seg[::-1], x_seg[::-1])

#         running_min = min(running_min, flow_vals[fmin_index])

#         for k in range(nf):
#             fv_idx = fmax_index + k       # index into flow_vals -> MRC unknown index
#             mrc_col = (numsegments - 1) + fv_idx  # column for MRC time unknown

#             if i == 0:
#                 # First segment: lag fixed to 0, equation: mrc_time[fv_idx] = interp
#                 rows_list.append(row)
#                 cols_list.append(mrc_col)
#                 vals_list.append(-1.0)
#                 b_list.append(interp_vals[k])
#             else:
#                 # lag_i + mrc_time[fv_idx] = interp  =>  lag_i - mrc = -interp
#                 lag_col = i - 1           # lag unknown column (0-indexed)
#                 rows_list.append(row)
#                 cols_list.append(lag_col)
#                 vals_list.append(1.0)

#                 rows_list.append(row)
#                 cols_list.append(mrc_col)
#                 vals_list.append(-1.0)

#                 b_list.append(interp_vals[k])
#             row += 1

#     if row == 0:
#         return np.array([[np.nan, np.nan]])

#     # Build sparse matrix and solve
#     A = lil_matrix((row, n_unknowns))
#     for r, c, v in zip(rows_list, cols_list, vals_list):
#         A[r, c] += v
#     A = A.tocsr()
#     b = np.array(b_list)

#     result = lsqr(A, -b)
#     solution = result[0]

#     lags = np.concatenate([[0.0], solution[:numsegments - 1]])
#     mrc_time = solution[numsegments - 1:]
#     mrc_time = np.sort(mrc_time)

#     # Shift so MRC starts at t=0
#     offset = mrc_time.min()
#     mrc_time -= offset

#     MRC = np.column_stack([mrc_time, flow_vals])
#     return MRC

def util_MasterRecessionCurve(Q, flow_section,
                               fit_method='nonparametric_analytic',
                               match_method='log'):
    Q = np.asarray(Q, dtype=float)
    flow_section = np.asarray(flow_section, dtype=int)

    if fit_method != 'nonparametric_analytic':
        raise NotImplementedError(f"fit_method '{fit_method}' is not implemented.")

    jitter_size = 1e-8
    numflows = 500
    rng = np.random.default_rng(0)

    numsegments = len(flow_section)
    flow_init_value = Q[flow_section[:, 0]]
    sortind = np.argsort(flow_init_value)[::-1]
    running_min = float(np.max(flow_init_value))

    # Build segments (descending flow, with jitter)
    segments = []
    for i in range(numsegments):
        seg = Q[flow_section[sortind[i], 0]: flow_section[sortind[i], 1] + 1].copy()
        seg[1:] += rng.normal(0, jitter_size, len(seg) - 1)
        seg = np.abs(seg) + 1e-20
        seg = np.sort(seg)[::-1]
        segments.append(seg)

    max_flow = max(s.max() for s in segments)
    min_flow = min(s.min() for s in segments)
    if min_flow <= 0:
        min_flow = jitter_size

    if match_method == 'linear':
        flow_vals = np.linspace(max_flow, min_flow, numflows)
    elif match_method == 'log':
        frac_log = 0.2
        gridspace = (max_flow - min_flow) / numflows
        linear_part = np.linspace(max_flow - gridspace / 2,
                                   min_flow + gridspace / 2,
                                   numflows - int(frac_log * numflows))
        log_part = np.logspace(np.log10(max_flow), np.log10(min_flow),
                                int(frac_log * numflows))
        flow_vals = np.sort(np.unique(np.concatenate([linear_part, log_part])))[::-1]
        flow_vals[-1] = min_flow
        flow_vals[0] = max_flow
        numflows = len(flow_vals)
    else:
        raise ValueError(f"match_method must be 'linear' or 'log'")

    # --- First pass: remove short segments ---
    short_segs = []
    for i, seg in enumerate(segments):
        fmax_index = np.searchsorted(-flow_vals, -seg[0], side='left')
        fmin_index = numflows - 1 if seg[-1] <= flow_vals[-1] else \
                     np.searchsorted(-flow_vals, -seg[-1], side='left') - 1
        if fmin_index - fmax_index + 1 <= 1:
            short_segs.append(i)

    for i in sorted(short_segs, reverse=True):
        segments.pop(i)

    if len(segments) == 0:
        return np.array([[np.nan, np.nan]])

    numsegments = len(segments)
    running_min = max(s[0] for s in segments)

    # Recompute flow_vals after removal
    max_flow = max(s.max() for s in segments)
    min_flow = min(s.min() for s in segments)
    flow_vals = flow_vals[(flow_vals <= max_flow) & (flow_vals >= min_flow)]
    numflows = len(flow_vals)

    # --- Build sparse system matching MATLAB exactly ---
    # MATLAB columns: [1..numsegments-1 = lags, numsegments..numsegments+numflows-1 = mrc_times]
    # In 0-indexed Python: lag_i = i-1 (for i>=1), mrc_col = (numsegments-1) + fv_idx
    
    rows_list, cols_list, vals_list, b_list = [], [], [], []
    row = 0
    bad_segs = []

    for i, seg in enumerate(segments):
        # Extend upward to running_min if gap exists (matches MATLAB)
        if seg[0] < running_min:
            seg = np.concatenate([[running_min], seg])

        fmax_index = np.searchsorted(-flow_vals, -seg[0], side='left')
        fmin_index = numflows - 1 if seg[-1] <= flow_vals[-1] else \
                     np.searchsorted(-flow_vals, -seg[-1], side='left') - 1
        nf = fmin_index - fmax_index + 1

        if nf == 0:
            bad_segs.append(i)
            continue

        # KEY FIX: match MATLAB's interp1(segment, 1:numel(segment), flow_vals(...))
        # MATLAB: X=segment (descending flow), V=1:numel(segment) (time indices 1-based)
        # i.e., given a flow value, find the corresponding time index
        x_seg = np.arange(1, len(seg) + 1, dtype=float)  # 1-based like MATLAB
        # seg is descending, so for interp we need ascending x -> flip both
        interp_vals = np.interp(
            flow_vals[fmax_index: fmin_index + 1],  # query points (flow values)
            seg[::-1],   # xp: ascending flow values
            x_seg[::-1]  # fp: corresponding time indices
        )

        running_min = min(running_min, flow_vals[fmin_index])

        for k in range(nf):
            fv_idx = fmax_index + k
            mrc_col = (numsegments - 1) + fv_idx

            if i == 0:
                # First segment: lag=0, equation: -mrc_time = interp_val  (matches MATLAB)
                rows_list.append(row)
                cols_list.append(mrc_col)
                vals_list.append(-1.0)
                b_list.append(interp_vals[k])
            else:
                lag_col = i - 1
                rows_list.append(row); cols_list.append(lag_col);  vals_list.append(1.0)
                rows_list.append(row); cols_list.append(mrc_col);  vals_list.append(-1.0)
                b_list.append(interp_vals[k])
            row += 1

    if row == 0:
        return np.array([[np.nan, np.nan]])

    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr

    n_unknowns = (numsegments - 1) + numflows
    A = lil_matrix((row, n_unknowns))
    for r, c, v in zip(rows_list, cols_list, vals_list):
        A[r, c] += v
    A = A.tocsr()
    b = np.array(b_list)

    solution = lsqr(A, -b)[0]

    lags = np.concatenate([[0.0], solution[:numsegments - 1]])
    mrc_time = solution[numsegments - 1:]
    mrc_time = np.sort(mrc_time)

    # Shift to start at 0 (matches MATLAB: mrc_time = mrc_time - offset; lags = lags - offset)
    offset = mrc_time.min()
    mrc_time -= offset
    lags = lags - offset  # <-- add this

    MRC = np.column_stack([mrc_time, flow_vals])
    # return MRC
    # In util_MasterRecessionCurve, change the return to also give back lags and seg indices
    return MRC, lags, [s for s in range(numsegments) if s not in bad_segs], segments


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def sig_BaseflowRecessionK(Q, t,
                            recession_length=15,
                            n_start=0,
                            eps=0.0,
                            start_of_recession='baseflow',
                            fit_method='nonparametric_analytic',
                            filter_par=0.925):
    """
    Calculate the baseflow recession constant K.

    Assumes exponential recession: Q(t) = Q0 * exp(-K * t).
    K < 0.065  -> groundwater-dominated, slow-draining system.
    K >= 0.065 -> shallow subsurface flow dominated, fast-draining system.

    Parameters
    ----------
    Q : array-like
        Streamflow [mm/timestep].
    t : array-like
        Time array (datetime64 or numeric).
    recession_length : int
        Minimum recession length in days (default 15).
    n_start : int
        Days to remove after recession start (default 0).
    eps : float
        Allowed flow increase during recession (default 0).
    start_of_recession : str
        'baseflow' or 'peak' (default 'baseflow').
    fit_method : str
        MRC fitting method (default 'nonparametric_analytic').
    filter_par : float
        Lyne-Hollick filter parameter (default 0.925).

    Returns
    -------
    BaseflowRecessionK : float
        Recession constant [1/timestep], or NaN on failure.
    error_flag : int
        0 = no error, 1 = warning, 2 = data error, 3 = calculation error.
    error_str : str
        Description of any error/warning.
    """
    Q = np.asarray(Q, dtype=float)
    t = np.asarray(t)

    # --- Data checks ---
    # MATLAB: [error_flag, error_str, timestep, t] = util_DataCheck(Q, t)
    # MATLAB: if error_flag == 2 -> return NaN
    error_flag, error_str, timestep, t = util_DataCheck(Q, t)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # Identify recession segments
    flow_section, ef, es = util_RecessionSegments(
        Q, t,
        recession_length=recession_length,
        n_start=n_start,
        eps=eps,
        start_of_recession=start_of_recession,
        filter_par=filter_par
    )
    if ef == 3:
        return np.nan, 3, es
    error_flag = max(error_flag, ef)
    error_str = error_str + es

    # # Build Master Recession Curve
    # MRC = util_MasterRecessionCurve(Q, flow_section,
    #                                  fit_method=fit_method,
    #                                  match_method='log')

    # if np.any(np.isnan(MRC)):
    #     return np.nan, 3, 'Error: MRC could not be constructed. ' + error_str

    # AFTER:
    MRC, _, _, _ = util_MasterRecessionCurve(Q, flow_section,
                                     fit_method=fit_method,
                                     match_method='log')

    if np.any(np.isnan(MRC[:, 0])):
        return np.nan, 3, 'Error: MRC could not be constructed. ' + error_str

    # Fit log-linear model: log(Q) = a + b*t  =>  K = -b
    mrc_t = MRC[:, 0]
    mrc_q = MRC[:, 1]

    valid = mrc_q > 0

    # fit_start_time = 15.0
    # valid = (mrc_q > 0) & (mrc_t >= fit_start_time)

    if valid.sum() < 2:
        return np.nan, 3, 'Error: Not enough valid MRC points for fitting. ' + error_str

    A = np.column_stack([np.ones(valid.sum()), mrc_t[valid]])
    mdl, _, _, _ = np.linalg.lstsq(A, np.log(mrc_q[valid]), rcond=None)

    BaseflowRecessionK = -mdl[1]

    if not np.isreal(BaseflowRecessionK):
        return np.nan, 3, 'Error: Complex BaseflowRecessionK. ' + error_str

    return float(BaseflowRecessionK), error_flag, error_str


# # ---------------------------------------------------------------------------
# # Example usage
# # ---------------------------------------------------------------------------
# if __name__ == '__main__':
#     # Synthetic example: exponential decay with noise
#     np.random.seed(42)
#     t = np.arange(0, 3650, dtype=float)          # 10 years of daily data
#     K_true = 0.05
#     Q = np.exp(-K_true * (t % 365)) + np.random.normal(0, 0.005, len(t))
#     Q = np.abs(Q)

#     t_dt = np.array(['2010-01-01'], dtype='datetime64[D]') + t.astype('timedelta64[D]')

#     K, flag, msg = sig_BaseflowRecessionK(Q, t_dt)
#     print(f"BaseflowRecessionK = {K:.4f}")
#     print(f"error_flag = {flag}")
#     if msg:
#         print(f"error_str  = {msg}")