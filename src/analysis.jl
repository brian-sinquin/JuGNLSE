"""
Pulse analysis functions for JuGNLSE.
"""

# ============================================================================
# Internal Helpers
# ============================================================================

"""
    _find_width_at_level(data::AbstractVector, grid::AbstractVector, level::Real)

Find the width of a distribution at a specific level relative to its peak.
Handles non-monotonic grids (like FFT-ordered omega) by sorting internally.
"""
function _find_width_at_level(data::AbstractVector, grid::AbstractVector, level::Real)
    # Handle non-monotonic grids (like FFT-ordered omega)
    if !issorted(grid)
        p = sortperm(grid)
        grid_sorted = grid[p]
        data_sorted = data[p]
    else
        grid_sorted = grid
        data_sorted = data
    end

    max_val = maximum(data_sorted)
    threshold = level * max_val

    # Find indices where data crosses threshold
    # We look for the first and last crossings
    indices = findall(data_sorted .>= threshold)

    if isempty(indices) || length(indices) < 2
        return 0.0
    end

    idx1 = indices[1]
    idx2 = indices[end]

    # Linear interpolation for better accuracy
    # Left side
    if idx1 > 1
        x0, x1 = grid_sorted[idx1 - 1], grid_sorted[idx1]
        y0, y1 = data_sorted[idx1 - 1], data_sorted[idx1]
        t_left = x0 + (x1 - x0) * (threshold - y0) / (y1 - y0)
    else
        t_left = grid_sorted[idx1]
    end

    # Right side
    if idx2 < length(grid_sorted)
        x0, x1 = grid_sorted[idx2], grid_sorted[idx2 + 1]
        y0, y1 = data_sorted[idx2], data_sorted[idx2 + 1]
        t_right = x0 + (x1 - x0) * (threshold - y0) / (y1 - y0)
    else
        t_right = grid_sorted[idx2]
    end

    return abs(t_right - t_left)
end

# ============================================================================
# Pulse Analysis Functions
# ============================================================================

"""
    pulse_energy(At::AbstractVector, dt::Real)

Calculate total pulse energy [J].
E = ∫ |A(t)|² dt
"""
function pulse_energy(At::AbstractVector, dt::Real)
    return sum(abs2, At) * dt
end

"""
    peak_power(At::AbstractVector)

Calculate peak power [W].
P_peak = max(|A(t)|²)
"""
function peak_power(At::AbstractVector)
    return maximum(abs2, At)
end

"""
    spectral_bandwidth(Aw::AbstractVector, omega::AbstractVector; level::Real=0.5)

Calculate spectral bandwidth at a given level (default 0.5 for FWHM) [rad/s].
"""
function spectral_bandwidth(Aw::AbstractVector, omega::AbstractVector; level::Real=0.5)
    return _find_width_at_level(abs2.(Aw), omega, level)
end

"""
    time_bandwidth_product(At::AbstractVector, Aw::AbstractVector, t::AbstractVector, omega::AbstractVector)

Calculate the Time-Bandwidth Product (TBP) using FWHM.
TBP = (Δt_fwhm * Δω_fwhm) / (4π)
"""
function time_bandwidth_product(
    At::AbstractVector, Aw::AbstractVector, t::AbstractVector, omega::AbstractVector
)
    dt_fwhm = _find_width_at_level(abs2.(At), t, 0.5)
    dw_fwhm = _find_width_at_level(abs2.(Aw), omega, 0.5)

    if isnan(dt_fwhm) || isnan(dw_fwhm)
        return NaN
    end

    return dt_fwhm * dw_fwhm / (4π)
end

"""
    fwhm(data::AbstractVector, grid::AbstractVector)

Calculate Full Width at Half Maximum (FWHM) of a distribution.
"""
function fwhm(data::AbstractVector, grid::AbstractVector)
    return _find_width_at_level(data, grid, 0.5)
end
