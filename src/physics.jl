"""
Physics-related utility functions and constants for JuGNLSE.
"""

# ============================================================================
# Physical and Mathematical Constants
# ============================================================================

"""
    SECH_FWHM_TO_T0

Conversion factor from FWHM to 1/e half-width for sech² pulse: 2 asinh(1) ≈ 1.7627.
"""
const SECH_FWHM_TO_T0 = 2 * asinh(1)  # ≈ 1.7627

"""
    GAUSSIAN_FWHM_TO_T0

Conversion factor from FWHM to 1/e half-width for Gaussian pulse: 2√(ln 2) ≈ 1.6651.
"""
const GAUSSIAN_FWHM_TO_T0 = 2 * sqrt(log(2))  # ≈ 1.6651

# ============================================================================
# Unit Conversion Functions
# ============================================================================

"""
    convert_loss(value, from::Tuple{Symbol,Symbol}, to::Tuple{Symbol,Symbol})

Convert loss coefficient between unit systems. Conversions: α[Np] = α[dB]·ln(10)/10.

  - `value`: Loss coefficient
  - `from`: (scale, length) where scale is `:dB` or `:linear`, length is `:km`, `:m`, `:cm`, `:mm`
  - `to`: Target units in same format

Returns converted loss value.
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

Convert decibels to linear scale: 10^(dB/10).
"""
db_to_linear(db::Real) = 10^(db / 10)

"""
    linear_to_db(linear::Real)

Convert linear scale to decibels: 10 log₁₀(value).
"""
linear_to_db(linear::Real) = 10 * log10(linear)

"""
    wavelength_to_frequency(lambda::Real)

Convert wavelength [m] to frequency [Hz]: ν = c/λ.
"""
wavelength_to_frequency(lambda::Real) = SPEED_OF_LIGHT / lambda

"""
    frequency_to_wavelength(freq::Real)

Convert frequency [Hz] to wavelength [m]: λ = c/ν.
"""
frequency_to_wavelength(freq::Real) = SPEED_OF_LIGHT / freq

# ============================================================================
# Soliton Formulas
# ============================================================================

"""
    calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)

Peak power [W] for fundamental (N=1) soliton: P₀ = |β₂|/(γT₀²).

  - `beta2`: Second-order dispersion [s²/m]
  - `gamma`: Nonlinear coefficient [1/(W·m)]
  - `T0`: Pulse duration (1/e half-width) [s]
"""
function calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)
    abs(beta2) / (gamma * T0^2)
end

"""
    soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)

Soliton order from pulse parameters: N² = (γP₀T₀²)/|β₂|.

  - `P_peak`: Peak power [W]
  - `beta2`: Second-order dispersion [s²/m]
  - `gamma`: Nonlinear coefficient [1/(W·m)]
  - `T0`: Pulse duration (1/e half-width) [s]
"""
function soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)
    sqrt(gamma * P_peak * T0^2 / abs(beta2))
end

"""
    dispersion_length(T0::Real, beta2::Real)

Dispersion length [m]: L_D = T₀²/|β₂|.

  - `T0`: Pulse duration (1/e half-width) [s]
  - `beta2`: Second-order dispersion [s²/m]
"""
function dispersion_length(T0::Real, beta2::Real)
    T0^2 / abs(beta2)
end

"""
    nonlinear_length(gamma::Real, P0::Real)

Nonlinear length [m]: L_NL = 1/(γP₀).

  - `gamma`: Nonlinear coefficient [1/(W·m)]
  - `P0`: Peak power [W]
"""
function nonlinear_length(gamma::Real, P0::Real)
    1.0 / (gamma * P0)
end

"""
    soliton_period(T0::Real, beta2::Real)

Fundamental soliton period [m]: z₀ = (π/2)L_D = (π/2)T₀²/|β₂|.

  - `T0`: Pulse duration (1/e half-width) [s]
  - `beta2`: Second-order dispersion [s²/m]
"""
function soliton_period(T0::Real, beta2::Real)
    π / 2 * dispersion_length(T0, beta2)
end

# ============================================================================
# Nonlinear coefficient construction
# ============================================================================

"""
    gamma_from_aeff(lambda0::Real, n2::Real, Aeff::Real)

Nonlinear coefficient [1/(W·m)] from material properties: γ = ω₀n₂/(cA_eff).

  - `lambda0`: Center wavelength [m]
  - `n2`: Nonlinear refractive index [m²/W]
  - `Aeff`: Effective mode area [m²]
"""
function gamma_from_aeff(lambda0::Real, n2::Real, Aeff::Real)
    omega0 = 2π * SPEED_OF_LIGHT / lambda0
    return omega0 * n2 / (SPEED_OF_LIGHT * Aeff)
end

"""
    gamma_from_aeff_vec(grid::Grid, n2::Real, Aeff_omega::AbstractVector)

Frequency-dependent nonlinear coefficient for M-GNLSE: γ(ω) = (ω₀+ω)n₂/(cA_eff(ω)).

  - `grid`: Grid object defining frequency array
  - `n2`: Nonlinear refractive index [m²/W]
  - `Aeff_omega`: Effective area at each grid frequency [m²]
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

Interpolate measured A_eff(λ) data to grid frequencies. Requires Interpolations.jl.

  - `grid`: Grid object defining target frequencies
  - `lambda_data`: Measured wavelengths [m]
  - `Aeff_data`: Measured effective areas [m²]
  - `extrapolation`: `:linear`, `:constant`, or `:error`
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
