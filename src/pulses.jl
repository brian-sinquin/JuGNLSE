"""
    sech_pulse(grid::Grid, width::Real, power_peak::Real, center_wavelength::Real;
               T0::Bool=false, time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)

Generate hyperbolic secant pulse A(t) = √P₀ sech(t/T₀) exp(-iC(t/T₀)²/2).

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `width::Real`: Pulse width [s] - FWHM if T0=false, 1/e half-width if T0=true
  - `power_peak::Real`: Peak power [W]
  - `center_wavelength::Real`: Center wavelength [m]
  - `T0::Bool`: Width interpretation flag (false=FWHM, true=1/e width)
  - `time_offset::Real`: Time offset from center [s]
  - `phase::Real`: Initial phase [rad]
  - `chirp::Real`: Linear chirp parameter C (0=transform-limited)
"""
function sech_pulse(
    grid::Grid,
    duration::Real,
    power_peak::Real;
    T0::Bool=false,
    time_offset::Real=0.0,
    phase::Real=0.0,
    chirp::Real=0.0,
)
    duration > 0 || throw(ArgumentError("Duration must be positive"))
    power_peak >= 0 || throw(ArgumentError("Peak power must be non-negative"))

    # Convert to 1/e half-width T0
    T0_param = T0 ? duration : duration / SECH_FWHM_TO_T0

    # Time array shifted by offset
    t_shifted = grid.t .- time_offset

    # Create sech envelope with fused broadcast for zero allocations
    # A(t) = √P₀ sech(t/T₀) exp(iφ) exp(-iC(t/T₀)²/2)
    At = similar(grid.t, ComplexF64)
    @. At =
        sqrt(power_peak) *
        sech(t_shifted / T0_param) *
        exp(im * phase) *
        exp(-im * chirp * (t_shifted / T0_param)^2 / 2)

    # Transform to frequency domain (using ifftshift to center pulse in FFT window)
    Aw = fft(ifftshift(At))

    Pulse(At, Aw, grid)
end

"""
    gaussian_pulse(grid::Grid, duration::Real, power_peak::Real;
                   T0::Bool=false, time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)

Generate Gaussian pulse A(t) = √P₀ exp(-(t/T₀)²/2) exp(-iC(t/T₀)²/2).

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `duration::Real`: Pulse duration [s] - FWHM if T0=false, 1/e half-width if T0=true
  - `power_peak::Real`: Peak power [W]
  - `T0::Bool`: Width interpretation flag (false=FWHM, true=1/e width)
  - `time_offset::Real`: Time offset from center [s]
  - `phase::Real`: Initial phase [rad]
  - `chirp::Real`: Linear chirp parameter C
"""
function gaussian_pulse(
    grid::Grid,
    duration::Real,
    power_peak::Real;
    T0::Bool=false,
    time_offset::Real=0.0,
    phase::Real=0.0,
    chirp::Real=0.0,
)
    duration > 0 || throw(ArgumentError("Duration must be positive"))
    power_peak >= 0 || throw(ArgumentError("Peak power must be non-negative"))

    # Convert duration to 1/e half-width T0
    T0_param = T0 ? duration : duration / GAUSSIAN_FWHM_TO_T0

    # Time array shifted by offset
    t_shifted = grid.t .- time_offset

    # Create Gaussian envelope with fused broadcast
    # A(t) = √P₀ exp(-(t/T₀)²/2) exp(iφ) exp(-iC(t/T₀)²/2)
    At = similar(grid.t, ComplexF64)
    @. At =
        sqrt(power_peak) *
        exp(-(t_shifted / T0_param)^2 / 2) *
        exp(im * phase) *
        exp(-im * chirp * (t_shifted / T0_param)^2 / 2)

    # Transform to frequency domain (using ifftshift to center pulse in FFT window)
    Aw = fft(ifftshift(At))

    Pulse(At, Aw, grid)
end

"""
    cw_pulse(grid::Grid, power::Real; phase::Real=0.0)

Generate continuous wave pulse with constant amplitude A(t) = √P exp(iφ).

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `power::Real`: CW power [W]
  - `phase::Real`: Initial phase [rad]
"""
function cw_pulse(grid::Grid, power::Real; phase::Real=0.0)
    power >= 0 || throw(ArgumentError("Power must be non-negative"))

    # Constant amplitude
    At = fill(sqrt(power) * exp(im * phase), grid.N)

    # Transform to frequency domain
    Aw = fft(At)

    Pulse(At, Aw, grid)
end

"""
    custom_pulse(grid::Grid, At::Vector{<:Complex})

Create pulse from user-defined time-domain envelope. Computes frequency representation via FFT.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `At::Vector{<:Complex}`: Time-domain envelope [√W] (length must match grid.N)
"""
function custom_pulse(grid::Grid, At::Vector{<:Complex})
    length(At) == grid.N || throw(ArgumentError("At length must match grid size"))

    # Transform to frequency domain
    Aw = fft(At)

    Pulse(copy(At), Aw, grid)
end
