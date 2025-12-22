"""
Post-simulation analysis functions for optical pulse characterization.

Provides energy, power, spectral bandwidth, and time-bandwidth product
calculations for Results objects and pulse envelopes.
"""

# ============================================================================
# Internal Helpers
# ============================================================================

"""
    _find_width_at_level(data::AbstractVector, grid::AbstractVector, level::Real)

Internal helper to compute distribution width at specified fractional level of peak.

Sorts non-monotonic grids (FFT-ordered frequencies) before analysis. Employs
linear interpolation at threshold crossings for sub-grid accuracy. Returns zero
if distribution does not reach specified level.
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

Compute total pulse energy [J] via temporal integration.

Performs numerical integration E = ∫ |A(t)|² dt using trapezoidal rule
with uniform time step `dt` [s]. Envelope normalization follows convention
|A(t)|² = instantaneous power [W].
"""
function pulse_energy(At::AbstractVector, dt::Real)
    return sum(abs2, At) * dt
end

"""
    peak_power(At::AbstractVector)

Extract peak instantaneous power [W] from temporal envelope.

Returns maximum value of |A(t)|² across entire time window. Assumes
envelope normalization where |A(t)|² represents instantaneous power.
"""
function peak_power(At::AbstractVector)
    return maximum(abs2, At)
end

"""
    spectral_bandwidth(Aw::AbstractVector, omega::AbstractVector; level::Real=0.5)

Compute spectral bandwidth [rad/s] at specified fractional intensity level.

Default `level=0.5` yields full-width at half-maximum (FWHM) of spectral
intensity |Ã(ω)|². Handles FFT-ordered frequency arrays automatically.
"""
function spectral_bandwidth(Aw::AbstractVector, omega::AbstractVector; level::Real=0.5)
    return _find_width_at_level(abs2.(Aw), omega, level)
end

"""
    time_bandwidth_product(At::AbstractVector, Aw::AbstractVector, t::AbstractVector, omega::AbstractVector)

Calculate dimensionless time-bandwidth product (TBP) from FWHM measurements.

Computes TBP = (Δt_FWHM · Δω_FWHM) / (4π), providing pulse quality metric.
Transform-limited pulses achieve minimum TBP for their shape (0.441 for
sech², 0.315 for Gaussian). Returns NaN if widths undefined.
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

Compute full-width at half-maximum (FWHM) of arbitrary distribution.

Generic interface to width calculation at 50% peak level. Handles both
temporal and spectral distributions with appropriate grid arrays.
"""
function fwhm(data::AbstractVector, grid::AbstractVector)
    return _find_width_at_level(data, grid, 0.5)
end
