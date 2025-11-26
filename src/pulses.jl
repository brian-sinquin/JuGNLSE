"""
    sech_pulse(grid::Grid, FWHM::Real, power_peak::Real, center_wavelength::Real; 
               time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)

Create a hyperbolic secant pulse.

# Arguments
- `grid::Grid`: Time-frequency grid
- `FWHM::Real`: Full-width at half-maximum [s]
- `power_peak::Real`: Peak power [W]
- `center_wavelength::Real`: Center wavelength [m]
- `time_offset::Real`: Time offset from center [s] (default: 0.0)
- `phase::Real`: Initial phase [rad] (default: 0.0)
- `chirp::Real`: Linear chirp parameter (default: 0.0)

# Returns
- `Pulse`: Pulse structure

# Example
```julia
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)  # 50 fs, 10 kW peak power
```
"""
function sech_pulse(grid::Grid, FWHM::Real, power_peak::Real, center_wavelength::Real;
                   time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)
    
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))
    power_peak >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    
    # Convert FWHM to 1/e half-width for sech
    # For intensity I(t) = I₀ sech²(t/T₀), FWHM occurs when sech²(t/T₀) = 0.5
    # This gives: sech(t_FWHM/T₀) = 1/sqrt(2), so t_FWHM = T₀ * acosh(sqrt(2))
    # Note: acosh(sqrt(2)) = asinh(1) ≈ 0.8814
    # Therefore: FWHM = 2 * T₀ * asinh(1)
    T0 = FWHM / (2 * asinh(1))

    # Time array shifted by offset
    t_shifted = grid.t .- time_offset
    
    # Create sech envelope: A(t) = sqrt(I₀) * sech(t/T₀)
    At = sqrt(power_peak) .* sech.(t_shifted ./ T0) .* 
         exp.(im .* phase) .* 
         exp.(-im .* chirp .* (t_shifted ./ T0).^2 ./ 2)
    
    # Transform to frequency domain
    Aw = ifft(At)
    
    Pulse(At, Aw, grid)
end

"""
    gaussian_pulse(grid::Grid, FWHM::Real, power_peak::Real, center_wavelength::Real;
                   time_offset::Real=0.0, phase::Real=0.0, chirp::Real=0.0)

Create a Gaussian pulse.

# Arguments
- `grid::Grid`: Time-frequency grid
- `FWHM::Real`: Full-width at half-maximum [s]
- `power_peak::Real`: Peak power [W]
- `center_wavelength::Real`: Center wavelength [m]
- `time_offset::Real`: Time offset from center [s] (default: 0.0)
- `phase::Real`: Initial phase [rad] (default: 0.0)
- `chirp::Real`: Linear chirp parameter (default: 0.0)

# Returns
- `Pulse`: Pulse structure
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

Create a continuous wave (CW) pulse.

# Arguments
- `grid::Grid`: Time-frequency grid
- `power::Real`: CW power [W]
- `phase::Real`: Initial phase [rad] (default: 0.0)

# Returns
- `Pulse`: Pulse structure
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
- `At::Vector{<:Complex}`: Time-domain envelope (must match grid size)

# Returns
- `Pulse`: Pulse structure
"""
function custom_pulse(grid::Grid, At::Vector{<:Complex})
    length(At) == grid.N || throw(ArgumentError("At length must match grid size"))
    
    # Transform to frequency domain
    Aw = ifft(At)
    
    Pulse(copy(At), Aw, grid)
end
