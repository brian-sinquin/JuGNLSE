"""
Utility functions for JuGNLSE package.
"""

"""
    calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)

Calculate the peak power for a fundamental soliton (N=1).

The soliton condition is: P₀ = |β₂| / (γ T₀²)

# Arguments
- `beta2::Real`: Second-order dispersion coefficient [ps²/m]
- `gamma::Real`: Nonlinear coefficient [W⁻¹m⁻¹]
- `T0::Real`: Pulse width parameter (1/e half-width) [s]

# Returns
- `Float64`: Peak power [W]

# Example
```julia
beta2 = -11.83e-27  # ps²/m
gamma = 0.11        # W⁻¹m⁻¹
FWHM = 50e-15       # 50 fs
T0 = FWHM / (2 * asinh(1))  # Convert FWHM to T₀
P0 = calculate_soliton_power(beta2, gamma, T0)
```
"""
function calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)
    abs(beta2) / (gamma * T0^2)
end

"""
    soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)

Calculate the soliton order N.

N² = γ P₀ T₀² / |β₂|

# Arguments
- `P_peak::Real`: Peak power [W]
- `beta2::Real`: Second-order dispersion coefficient [s²/m]
- `gamma::Real`: Nonlinear coefficient [W⁻¹m⁻¹]
- `T0::Real`: Pulse width parameter [s]

# Returns
- `Float64`: Soliton order N
"""
function soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)
    sqrt(gamma * P_peak * T0^2 / abs(beta2))
end

"""
    pulse_energy(At::Vector{<:Complex}, dt::Real)

Calculate total pulse energy.

E = ∫|A(t)|² dt

# Arguments
- `At::Vector{<:Complex}`: Time-domain pulse envelope
- `dt::Real`: Time step [s]

# Returns
- `Float64`: Pulse energy [J]
"""
function pulse_energy(At::Vector{<:Complex}, dt::Real)
    sum(abs2.(At)) * dt
end

"""
    peak_power(At::Vector{<:Complex})

Calculate peak power of a pulse.

# Arguments
- `At::Vector{<:Complex}`: Time-domain pulse envelope

# Returns
- `Float64`: Peak power [W]
"""
function peak_power(At::Vector{<:Complex})
    maximum(abs2.(At))
end

"""
    spectral_bandwidth(Aw::Vector{<:Complex}, omega::Vector{<:Real}; level::Real=0.5)

Calculate spectral bandwidth at a given level (default FWHM at 0.5).

# Arguments
- `Aw::Vector{<:Complex}`: Frequency-domain spectrum
- `omega::Vector{<:Real}`: Angular frequency grid [rad/s]
- `level::Real`: Fraction of peak for bandwidth calculation (default: 0.5 for FWHM)

# Returns
- `Float64`: Spectral bandwidth [rad/s]
"""
function spectral_bandwidth(Aw::Vector{<:Complex}, omega::Vector{<:Real}; level::Real=0.5)
    spectrum = abs2.(Aw)
    max_val = maximum(spectrum)
    threshold = level * max_val
    
    # Find indices where spectrum exceeds threshold
    above_threshold = spectrum .>= threshold
    indices = findall(above_threshold)
    
    if isempty(indices)
        return 0.0
    end
    
    omega_min = omega[minimum(indices)]
    omega_max = omega[maximum(indices)]
    
    omega_max - omega_min
end

"""
    time_bandwidth_product(At::Vector{<:Complex}, Aw::Vector{<:Complex}, 
                          t::Vector{<:Real}, omega::Vector{<:Real})

Calculate time-bandwidth product (TBP).

For transform-limited pulses:
- Gaussian: TBP = 0.441
- Sech: TBP = 0.315

# Arguments
- `At::Vector{<:Complex}`: Time-domain envelope
- `Aw::Vector{<:Complex}`: Frequency-domain spectrum
- `t::Vector{<:Real}`: Time grid
- `omega::Vector{<:Real}`: Frequency grid

# Returns
- `Float64`: Time-bandwidth product (dimensionless)
"""
function time_bandwidth_product(At::Vector{<:Complex}, Aw::Vector{<:Complex}, 
                               t::Vector{<:Real}, omega::Vector{<:Real})
    # Calculate temporal FWHM
    It = abs2.(At)
    max_It = maximum(It)
    threshold_t = 0.5 * max_It
    above_t = It .>= threshold_t
    indices_t = findall(above_t)
    
    if isempty(indices_t)
        return NaN
    end
    
    dt_fwhm = abs(t[maximum(indices_t)] - t[minimum(indices_t)])
    
    # Calculate spectral FWHM
    Iw = abs2.(Aw)
    max_Iw = maximum(Iw)
    threshold_w = 0.5 * max_Iw
    above_w = Iw .>= threshold_w
    indices_w = findall(above_w)
    
    if isempty(indices_w)
        return NaN
    end
    
    dw_fwhm = abs(omega[maximum(indices_w)] - omega[minimum(indices_w)])
    
    # TBP = Δt * Δω / (4π) or Δt * Δν (in frequency)
    dt_fwhm * dw_fwhm / (4π)
end

"""
    fwhm(data::Vector{<:Real}, grid::Vector{<:Real})

Calculate full-width at half-maximum of a distribution.

# Arguments
- `data::Vector{<:Real}`: Data values (e.g., intensity)
- `grid::Vector{<:Real}`: Grid points (e.g., time or frequency)

# Returns
- `Float64`: FWHM in units of grid
"""
function fwhm(data::Vector{<:Real}, grid::Vector{<:Real})
    max_val = maximum(data)
    threshold = 0.5 * max_val
    
    above = data .>= threshold
    indices = findall(above)
    
    if isempty(indices)
        return 0.0
    end
    
    abs(grid[maximum(indices)] - grid[minimum(indices)])
end

"""
    db_to_linear(db::Real)

Convert decibels to linear scale.

# Arguments
- `db::Real`: Value in decibels

# Returns
- `Float64`: Linear value
"""
db_to_linear(db::Real) = 10^(db/10)

"""
    linear_to_db(linear::Real)

Convert linear scale to decibels.

# Arguments
- `linear::Real`: Linear value

# Returns
- `Float64`: Value in decibels
"""
linear_to_db(linear::Real) = 10 * log10(linear)

"""
    wavelength_to_frequency(lambda::Real)

Convert wavelength to frequency.

# Arguments
- `lambda::Real`: Wavelength [m]

# Returns
- `Float64`: Frequency [Hz]
"""
wavelength_to_frequency(lambda::Real) = 3e8 / lambda

"""
    frequency_to_wavelength(freq::Real)

Convert frequency to wavelength.

# Arguments
- `freq::Real`: Frequency [Hz]

# Returns
- `Float64`: Wavelength [m]
"""
frequency_to_wavelength(freq::Real) = 3e8 / freq
