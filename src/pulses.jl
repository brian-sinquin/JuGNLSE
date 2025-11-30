"""
    sech_pulse(grid::Grid, width::Real, power_peak::Real, center_wavelength::Real; 
               T0::Bool=false, time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)

Create a hyperbolic secant pulse: A(t) = √P₀ sech(t/T₀).

# Arguments
- `grid::Grid`: Time-frequency grid
- `width::Real`: Pulse width [s] - interpretation depends on `T0` flag
- `power_peak::Real`: Peak power [W]
- `center_wavelength::Real`: Center wavelength [m]
- `T0::Bool`: If `true`, `width` is 1/e half-width T₀; if `false` (default), `width` is FWHM
- `time_offset::Real`: Time offset from center [s] (default: 0.0)
- `phase::Real`: Initial phase [rad] (default: 0.0)
- `chirp::Real`: Linear chirp parameter (default: 0.0)

# Returns
- `Pulse`: Pulse structure with time and frequency domain representations

# Width Parameter Interpretation
The `T0` flag controls how `width` is interpreted:

- **`T0=false` (default)**: `width` is the Full-Width at Half-Maximum (FWHM) of intensity
  * Intensity profile: I(t) = I₀ sech²(t/T₀) where T₀ = FWHM / (2·asinh(1)) ≈ FWHM / 1.763
  * FWHM occurs when sech²(t/T₀) = 0.5
  * Typical for experimental specifications: "50 fs pulse" usually means 50 fs FWHM

- **`T0=true`**: `width` is the 1/e amplitude half-width T₀
  * Amplitude profile: A(t) = A₀ sech(t/T₀)
  * **Use this for soliton calculations!** Soliton power P₀ = |β₂|/(γT₀²) requires T₀
  * Conversion: T₀ ≈ 0.567 × FWHM or FWHM ≈ 1.763 × T₀

# Examples
```julia
# Standard usage with FWHM (experimental specification)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)  
# 50 fs FWHM, 10 kW peak power, 835 nm center wavelength

# Soliton calculation (use T0 parameter!)
beta2 = -11.83e-27  # ps²/m  
gamma = 0.11        # W⁻¹m⁻¹
T0 = 28.4e-15       # 28.4 fs 1/e width
P0 = abs(beta2) / (gamma * T0^2)  # Soliton power
pulse = sech_pulse(grid, T0, P0, 835e-9, T0=true)  # N=1 soliton

# Chirped pulse
pulse = sech_pulse(grid, 100e-15, 5000.0, 1030e-9, chirp=2.0)
# Linear chirp: instantaneous frequency varies quadratically with time

# Delayed pulse
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9, time_offset=500e-15)
# Pulse centered at t = +500 fs
```

# Chirp Parameter
The chirp parameter C introduces quadratic phase modulation:
```
A(t) = √P₀ sech(t/T₀) exp(-iC(t/T₀)²/2)
```
- C = 0: Transform-limited (unchirped) pulse
- C > 0: Up-chirp (red-to-blue, positive frequency sweep)
- C < 0: Down-chirp (blue-to-red, negative frequency sweep)

# Notes
- Sech pulses have Time-Bandwidth Product (TBP) = 0.315 when transform-limited
- For chirped pulses: TBP = 0.315√(1 + C²)
- **Critical for solitons**: Always use `T0=true` when calculating soliton parameters!

# See Also
- [`gaussian_pulse`](@ref): Alternative pulse shape
- [`calculate_soliton_power`](@ref): Compute soliton power from parameters
- [`soliton_order`](@ref): Calculate soliton order N
"""
function sech_pulse(grid::Grid, width::Real, power_peak::Real, center_wavelength::Real;
                   T0::Bool=false, time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)
    
    width > 0 || throw(ArgumentError("Width must be positive"))
    power_peak >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    
    # Convert to 1/e half-width T0
    if T0
        # width is already T0
        T0_param = width
    else
        # width is FWHM, convert to T0
        # For intensity I(t) = I₀ sech²(t/T₀), FWHM occurs when sech²(t/T₀) = 0.5
        # This gives: sech(t_FWHM/T₀) = 1/sqrt(2), so t_FWHM = T₀ * acosh(sqrt(2))
        # Note: acosh(sqrt(2)) = asinh(1) ≈ 0.8814
        # Therefore: FWHM = 2 * T₀ * asinh(1)
        T0_param = width / (2 * asinh(1))
    end

    # Time array shifted by offset
    t_shifted = grid.t .- time_offset
    
    # Create sech envelope: A(t) = sqrt(I₀) * sech(t/T₀)
    At = sqrt(power_peak) .* sech.(t_shifted ./ T0_param) .* 
         exp.(im .* phase) .* 
         exp.(-im .* chirp .* (t_shifted ./ T0_param).^2 ./ 2)
    
    # Transform to frequency domain
    Aw = ifft(At)
    
    Pulse(At, Aw, grid)
end

"""
    gaussian_pulse(grid::Grid, FWHM::Real, power_peak::Real, center_wavelength::Real;
                   time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)

Create a Gaussian pulse: A(t) = √P₀ exp(-(t/T₀)²/2).

# Arguments
- `grid::Grid`: Time-frequency grid
- `FWHM::Real`: Full-width at half-maximum [s]
- `power_peak::Real`: Peak power [W]
- `center_wavelength::Real`: Center wavelength [m]
- `time_offset::Real`: Time offset from center [s] (default: 0.0)
- `phase::Real`: Initial phase [rad] (default: 0.0)
- `chirp::Real`: Linear chirp parameter (default: 0.0)

# Returns
- `Pulse`: Pulse structure with Gaussian time-domain profile

# Width Parameter
- `FWHM`: Intensity full-width at half-maximum where I(t) = I₀/2
- Conversion to 1/e amplitude width: T₀ = FWHM / (2√ln2) ≈ FWHM / 1.665
- Intensity profile: I(t) = I₀ exp(-(t/T₀)²)

# Examples
```julia
# Standard Gaussian pulse
pulse = gaussian_pulse(grid, 100e-15, 5000.0, 1030e-9)
# 100 fs FWHM, 5 kW peak, 1030 nm

# Chirped Gaussian
pulse = gaussian_pulse(grid, 200e-15, 2000.0, 1550e-9, chirp=-3.0)
# Down-chirped 200 fs pulse

# Off-center pulse
pulse = gaussian_pulse(grid, 150e-15, 8000.0, 835e-9, time_offset=1e-12)
# Pulse centered at t = +1 ps
```

# Chirp Parameter
Similar to sech pulses, chirp introduces quadratic phase:
```
A(t) = √P₀ exp(-(t/T₀)²/2) exp(-iC(t/T₀)²/2)
```
Combined effect: spectral broadening and temporal stretching/compression

# Notes
- Transform-limited Gaussian: TBP = 0.441 (higher than sech's 0.315)
- Gaussian pulses have no wings (decay faster than sech) → less nonlinear interaction
- For chirped pulses: TBP = 0.441√(1 + C²)
- Gaussian is exact eigenfunction of parabolic potential (harmonic oscillator)

# See Also
- [`sech_pulse`](@ref): Hyperbolic secant pulse (more common for solitons)
- [`time_bandwidth_product`](@ref): Calculate TBP to check pulse quality
"""
function gaussian_pulse(grid::Grid, FWHM::Real, power_peak::Real, center_wavelength::Real;
                       time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))
    power_peak >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    
    # Convert FWHM to 1/e half-width for Gaussian
    # For intensity I(t) = I₀ exp(-(t/T₀)²), FWHM occurs when exp(-(t/T₀)²) = 0.5
    # This gives: (t_FWHM/T₀)² = ln(2), so t_FWHM = T₀ * sqrt(ln(2))
    # Therefore: FWHM = 2 * T₀ * sqrt(ln(2))
    T0 = FWHM / (2 * sqrt(log(2)))
    
    # Time array shifted by offset
    t_shifted = grid.t .- time_offset
    
    # Create Gaussian envelope: A(t) = sqrt(I₀) * exp(-(t/T₀)²/2)
    At = sqrt(power_peak) .* exp.(-(t_shifted ./ T0).^2 ./ 2) .*
         exp.(im .* phase) .*
         exp.(-im .* chirp .* (t_shifted ./ T0).^2 ./ 2)
    
    # Transform to frequency domain
    Aw = ifft(At)
    
    Pulse(At, Aw, grid)
end

"""
    cw_pulse(grid::Grid, power::Real; phase::Real=0.0)

Create a continuous wave (CW) pulse with constant amplitude.

# Arguments
- `grid::Grid`: Time-frequency grid
- `power::Real`: CW power [W]
- `phase::Real`: Initial phase [rad] (default: 0.0)

# Returns
- `Pulse`: Pulse structure with flat temporal profile

# Example
```julia
# CW at 1 W
pulse = cw_pulse(grid, 1.0)

# CW with phase offset
pulse = cw_pulse(grid, 0.5, phase=π/4)
```

# Notes
- CW has all energy at center frequency (delta function in frequency domain)
- Useful for testing dispersion effects without nonlinearity
- For modulation instability studies: add small perturbation to CW
"""
function cw_pulse(grid::Grid, power::Real; phase::Real=0.0)
    power >= 0 || throw(ArgumentError("Power must be non-negative"))
    
    # Constant amplitude
    At = fill(sqrt(power) * exp(im * phase), grid.N)
    
    # Transform to frequency domain
    Aw = ifft(At)
    
    Pulse(At, Aw, grid)
end

"""
    custom_pulse(grid::Grid, At::Vector{<:Complex})

Create a pulse from a custom time-domain envelope.

# Arguments
- `grid::Grid`: Time-frequency grid
- `At::Vector{<:Complex}`: Time-domain envelope [√W] (must match grid size)

# Returns
- `Pulse`: Pulse structure

# Example
```julia
# Custom double-pulse
At = sech.(grid.t ./ 50e-15) .+ 0.5 .* sech.((grid.t .- 300e-15) ./ 50e-15)
At .*= sqrt(10000.0)  # Scale to 10 kW peak
pulse = custom_pulse(grid, At)

# Arbitrary modulated pulse
At = sqrt(5000.0) .* exp.(-(grid.t ./ 100e-15).^2) .* cos.(2π * 1e12 .* grid.t)
pulse = custom_pulse(grid, At)
```

# Notes
- Input `At` is copied to avoid external modifications
- Automatically computes frequency-domain representation via FFT
- Useful for complex pulse shapes, pulse trains, or modulated waveforms
"""
function custom_pulse(grid::Grid, At::Vector{<:Complex})
    length(At) == grid.N || throw(ArgumentError("At length must match grid size"))
    
    # Transform to frequency domain
    Aw = ifft(At)
    
    Pulse(copy(At), Aw, grid)
end
