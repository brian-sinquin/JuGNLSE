"""
Physics-related utility functions and constants for JuGNLSE.
"""

# ============================================================================
# Physical and Mathematical Constants
# ============================================================================

"""
Conversion factor: FWHM to T₀ (1/e half-width) for sech pulse  # ≈ 1.7627
"""
const SECH_FWHM_TO_T0 = 2 * asinh(1)  # ≈ 1.7627

"""
Conversion factor: FWHM to T₀ (1/e half-width) for Gaussian pulse  # ≈ 1.6651
"""
const GAUSSIAN_FWHM_TO_T0 = 2 * sqrt(log(2))  # ≈ 1.6651

# ============================================================================
# Unit Conversion Functions
# ============================================================================

"""
    convert_loss(value, from::Tuple{Symbol,Symbol}, to::Tuple{Symbol,Symbol})

Convert loss coefficient between different units.

# Arguments

  - `value`: Loss coefficient value
  - `from`: Tuple of (scale, length) where scale is `:dB` or `:linear` (Nepers)
    and length is `:km`, `:m`, `:cm`, or `:mm`
  - `to`: Target units in same format as `from`

# Returns

  - Converted loss value

# Examples

```julia
# dB/km to linear (Nepers/m)
alpha_npm = convert_loss(0.2, (:dB, :km), (:linear, :m))

# dB/cm to dB/km
alpha_dbkm = convert_loss(0.02, (:dB, :cm), (:dB, :km))

# Linear (Np/m) to dB/km
alpha_dbkm = convert_loss(4.6e-5, (:linear, :m), (:dB, :km))
```

# Notes

  - dB to linear: α[Np] = α[dB] × ln(10)/10
  - Linear to dB: α[dB] = α[Np] × 10/ln(10)
  - Length conversions: km = 1000m, cm = 0.01m, mm = 0.001m
"""
function convert_loss(value::Real, from::Tuple{Symbol, Symbol}, to::Tuple{Symbol, Symbol})
    from_scale, from_length = from
    to_scale, to_length = to

    # Length conversion factors to meters
    length_to_m = Dict(:km => 1000.0, :m => 1.0, :cm => 0.01, :mm => 0.001)

    haskey(length_to_m, from_length) ||
        throw(ArgumentError("Unknown length unit: $from_length (use :km, :m, :cm, or :mm)"))
    haskey(length_to_m, to_length) ||
        throw(ArgumentError("Unknown length unit: $to_length (use :km, :m, :cm, or :mm)"))

    # Step 1: Convert to linear scale if needed
    linear_value = if from_scale == :dB
        value * log(10) / 10  # dB to Nepers
    elseif from_scale == :linear
        value
    else
        throw(ArgumentError("Unknown loss scale: $from_scale (use :dB or :linear)"))
    end

    # Step 2: Convert length units (to per-meter basis)
    value_per_m = linear_value / length_to_m[from_length]

    # Step 3: Convert to target length units
    value_target_length = value_per_m * length_to_m[to_length]

    # Step 4: Convert to target scale
    result = if to_scale == :linear
        value_target_length
    elseif to_scale == :dB
        value_target_length * 10 / log(10)  # Nepers to dB
    else
        throw(ArgumentError("Unknown loss scale: $to_scale (use :dB or :linear)"))
    end

    return result
end

"""
    db_to_linear(db::Real)

Convert decibels to linear scale.
"""
db_to_linear(db::Real) = 10^(db / 10)

"""
    linear_to_db(linear::Real)

Convert linear scale to decibels.
"""
linear_to_db(linear::Real) = 10 * log10(linear)

"""
    wavelength_to_frequency(lambda::Real)

Convert wavelength to frequency [Hz].
"""
wavelength_to_frequency(lambda::Real) = SPEED_OF_LIGHT / lambda

"""
    frequency_to_wavelength(freq::Real)

Convert frequency [Hz] to wavelength [m].
"""
frequency_to_wavelength(freq::Real) = SPEED_OF_LIGHT / freq

# ============================================================================
# Soliton Formulas
# ============================================================================

"""
    calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)

Calculate the peak power required for a fundamental (N=1) soliton.
P₀ = |β₂| / (γ T₀²)
"""
function calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)
    abs(beta2) / (gamma * T0^2)
end

"""
    soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)

Calculate the soliton order N from pulse parameters.
N² = (γ P₀ T₀²) / |β₂|
"""
function soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)
    sqrt(gamma * P_peak * T0^2 / abs(beta2))
end

"""
    dispersion_length(T0::Real, beta2::Real)

Calculate the dispersion length L_D = T₀²/|β₂|.
"""
function dispersion_length(T0::Real, beta2::Real)
    T0^2 / abs(beta2)
end

"""
    nonlinear_length(gamma::Real, P0::Real)

Calculate the nonlinear length L_NL = 1/(γ·P₀).
"""
function nonlinear_length(gamma::Real, P0::Real)
    1.0 / (gamma * P0)
end

"""
    soliton_period(T0::Real, beta2::Real)

Calculate fundamental soliton period z₀ = π/2 · L_D.
"""
function soliton_period(T0::Real, beta2::Real)
    π / 2 * dispersion_length(T0, beta2)
end

# ============================================================================
# Nonlinear coefficient construction
# ============================================================================

"""
    gamma_from_aeff(lambda0::Real, n2::Real, Aeff::Real)

Compute nonlinear coefficient γ from material and fiber parameters (scalar).
γ = (2π/λ₀) × (n₂/Aeff)
"""
function gamma_from_aeff(lambda0::Real, n2::Real, Aeff::Real)
    omega0 = 2π * SPEED_OF_LIGHT / lambda0
    return omega0 * n2 / (SPEED_OF_LIGHT * Aeff)
end

"""
    gamma_from_aeff_vec(grid::Grid, n2::Real, Aeff_omega::AbstractVector)

Compute frequency-dependent nonlinear coefficient γ(ω) for M-GNLSE.
γ(ω) = (ω₀ + ω) × n₂ / (c × Aeff(ω))
"""
function gamma_from_aeff_vec(grid::Grid, n2::Real, Aeff_omega::AbstractVector)
    length(Aeff_omega) == grid.N || throw(
        ArgumentError("Aeff_omega must have length $(grid.N), got $(length(Aeff_omega))"),
    )
    omega0 = grid.omega0
    gamma_vec = (omega0 .+ grid.omega) .* n2 ./ (SPEED_OF_LIGHT .* Aeff_omega)
    return max.(gamma_vec, 0.0)
end

"""
    aeff_from_measured_data(grid::Grid, lambda_data::AbstractVector, Aeff_data::AbstractVector;
                           extrapolation::Symbol=:linear)

Interpolate measured effective area data to grid frequencies.
Requires `Interpolations.jl` to be loaded in the environment.
"""
function aeff_from_measured_data(
    grid::Grid,
    lambda_data::AbstractVector,
    Aeff_data::AbstractVector;
    extrapolation::Symbol=:linear,
)
    length(lambda_data) == length(Aeff_data) ||
        throw(ArgumentError("lambda_data and Aeff_data must have same length"))

    if !isdefined(Main, :Interpolations)
        error("Interpolations.jl package required but not loaded.")
    end

    omega_data = 2π .* SPEED_OF_LIGHT ./ lambda_data
    sorted_indices = sortperm(omega_data)
    omega_data_sorted = omega_data[sorted_indices]
    Aeff_data_sorted = Aeff_data[sorted_indices]

    extrap = if extrapolation == :linear
        Main.Interpolations.Line()
    elseif extrapolation == :constant
        Main.Interpolations.Flat()
    elseif extrapolation == :error
        throw(ArgumentError("Grid frequencies outside measured range"))
    else
        throw(ArgumentError("Unknown extrapolation method: $extrapolation"))
    end

    itp = Main.Interpolations.LinearInterpolation(
        omega_data_sorted, Aeff_data_sorted; extrapolation_bc=extrap
    )

    omega_grid = grid.omega0 .+ grid.omega
    return itp.(omega_grid)
end
