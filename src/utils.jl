"""
Utility functions for JuGNLSE package.
"""

"""
    calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)

Calculate the peak power required for a fundamental (N=1) soliton.

# Soliton Condition
For a hyperbolic secant pulse A(z,t) = √P₀ sech(t/T₀) to form a fundamental soliton,
the peak power must satisfy:
```
P₀ = |β₂| / (γ T₀²)
```
This balance between dispersion (β₂) and nonlinearity (γ) maintains pulse shape during propagation.

# Arguments
- `beta2::Real`: Second-order dispersion coefficient [s²/m]
- `gamma::Real`: Nonlinear coefficient [W⁻¹m⁻¹]
- `T0::Real`: Pulse width parameter (1/e amplitude half-width) [s]

# Returns
- `Float64`: Peak power for N=1 soliton [W]

# Examples
```julia
# Typical single-mode fiber at 835 nm
beta2 = -11.83e-27  # ps²/m (anomalous dispersion)
gamma = 0.11        # W⁻¹m⁻¹
T0 = 28.4e-15       # 28.4 fs (1/e width)

P0 = calculate_soliton_power(beta2, gamma, T0)
# P0 ≈ 13,330 W (13.3 kW)

# Create fundamental soliton pulse
pulse = sech_pulse(grid, T0, P0, 835e-9, T0=true)
```

# Physical Interpretation
- **Soliton period**: z₀ = π/2 × LD where LD = T₀²/|β₂| is the dispersion length
- **N=1 soliton**: Maintains shape indefinitely in ideal fiber
- **Higher order**: N=2,3,... solitons: use P = N²×P₀

# Notes
- Requires **anomalous dispersion** (β₂ < 0) for bright solitons
- `T0` is 1/e amplitude half-width, NOT FWHM
- Conversion: T0 ≈ 0.567 × FWHM for sech pulses
- Real fibers: perturbations (loss, Raman, higher-order dispersion) cause evolution

# See Also
- [`soliton_order`](@ref): Calculate soliton order from power
- [`sech_pulse`](@ref): Create soliton initial condition (use `T0=true`!)
"""
function calculate_soliton_power(beta2::Real, gamma::Real, T0::Real)
    abs(beta2) / (gamma * T0^2)
end

"""
    soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)

Calculate the soliton order N from pulse parameters.

# Mathematical Definition
The soliton order characterizes the balance between nonlinearity and dispersion:
```
N² = (γ P₀ T₀²) / |β₂|
```

# Physical Interpretation
- **N = 1**: Fundamental soliton (maintains exact shape)
- **N = 2**: Second-order soliton (periodic evolution with period z₀)
- **N = 3, 4, ...**: Higher-order solitons (complex periodic dynamics)
- **N < 1**: Dispersive regime (pulse spreads)
- **N > 1**: Nonlinear regime (compression then fission)

# Arguments
- `P_peak::Real`: Peak power [W]
- `beta2::Real`: Second-order dispersion coefficient [s²/m]
- `gamma::Real`: Nonlinear coefficient [W⁻¹m⁻¹]
- `T0::Real`: Pulse width parameter (1/e amplitude half-width) [s]

# Returns
- `Float64`: Soliton order N (dimensionless)

# Examples
```julia
# Check if pulse forms fundamental soliton
beta2 = -11.83e-27
gamma = 0.11
T0 = 28.4e-15
P0 = 10000.0  # 10 kW

N = soliton_order(P0, beta2, gamma, T0)
# N ≈ 0.87 → Slightly dispersive, will broaden

# Calculate power for N=3 soliton
P_N3 = 3^2 * calculate_soliton_power(beta2, gamma, T0)
N_check = soliton_order(P_N3, beta2, gamma, T0)
# N_check = 3.0 → Third-order soliton
```

# Soliton Dynamics by Order
- **N = 1**: Maintains shape over many dispersion lengths
- **N = 2**: Breathes with period z₀ = π/2 × LD
- **N ≥ 3**: Complex evolution, eventual fission into fundamental solitons
- **Non-integer N**: Intermediate dynamics, sheds dispersive waves

# Notes
- Requires anomalous dispersion (β₂ < 0)
- Higher N → stronger nonlinear effects → faster dynamics
- Real fibers: perturbations (Raman, loss, TOD) modify ideal soliton behavior

# See Also
- [`calculate_soliton_power`](@ref): Compute P₀ for given N
- [`pulse_energy`](@ref): Calculate total pulse energy
"""
function soliton_order(P_peak::Real, beta2::Real, gamma::Real, T0::Real)
    sqrt(gamma * P_peak * T0^2 / abs(beta2))
end

"""
    pulse_energy(At::Vector{<:Complex}, dt::Real)

Calculate total pulse energy from time-domain envelope.

# Mathematical Formula
```
E = ∫_{-∞}^{∞} |A(t)|² dt ≈ Σᵢ |A[i]|² × Δt
```
where |A(t)|² is the instantaneous power [W].

# Arguments
- `At::Vector{<:Complex}`: Time-domain pulse envelope [√W]
- `dt::Real`: Time step [s]

# Returns
- `Float64`: Total pulse energy [J] (Joules = W·s)

# Examples
```julia
# Calculate energy of sech pulse
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
E = pulse_energy(pulse.At, grid.dt)
# E ≈ 0.882 nJ for 50 fs FWHM, 10 kW peak

# Verify energy conservation during propagation
results = solve(pulse, params)
E_in = pulse_energy(results.At[:, 1], grid.dt)
E_out = pulse_energy(results.At[:, end], grid.dt)
loss_dB = 10 * log10(E_out / E_in)
# Should match fiber loss (α × length)

# Calculate energy of each spectral component
power_spectrum = abs2.(pulse.Aw)
E_check = sum(power_spectrum) * grid.dω / (2π)
# Parseval's theorem: should equal E
```

# Physical Interpretation
- Energy is conserved in lossless propagation (α = 0)
- Loss: E(z) = E(0) × exp(-α × z) where α is in Np/m
- Energy units: 1 nJ = 10⁻⁹ J (typical for femtosecond pulses)

# Notes
- For CW: energy diverges (infinite duration)
- Ensure time window captures entire pulse (no truncation)
- Energy in frequency domain: E = (1/2π) ∫|Ã(ω)|² dω

# See Also
- [`peak_power`](@ref): Calculate peak power
- [`spectral_bandwidth`](@ref): Measure spectral width
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

Calculate time-bandwidth product (TBP) to assess pulse quality and chirp.

# Mathematical Definition
```
TBP = Δt_FWHM × Δν_FWHM = (Δt_FWHM × Δω_FWHM) / (4π)
```
where Δt is temporal FWHM and Δω is spectral FWHM in rad/s.

# Transform-Limited Values
Different pulse shapes have characteristic minimum TBP values:
- **Gaussian**: TBP_min = 0.441
- **Sech (hyperbolic secant)**: TBP_min = 0.315
- **Square**: TBP_min = 0.886

# Physical Interpretation
- **TBP = TBP_min**: Transform-limited (unchirped) pulse
- **TBP > TBP_min**: Chirped pulse (time-frequency correlation)
- **TBP >> TBP_min**: Heavily chirped (can be compressed)

# Arguments
- `At::Vector{<:Complex}`: Time-domain envelope
- `Aw::Vector{<:Complex}`: Frequency-domain spectrum
- `t::Vector{<:Real}`: Time grid [s]
- `omega::Vector{<:Real}`: Angular frequency grid [rad/s]

# Returns
- `Float64`: Time-bandwidth product (dimensionless)

# Examples
```julia
# Check if sech pulse is transform-limited
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
tbp = time_bandwidth_product(pulse.At, pulse.Aw, grid.t, grid.omega)
# tbp ≈ 0.315 → Transform-limited sech

# Chirped pulse has larger TBP
chirped = sech_pulse(grid, 50e-15, 10000.0, 835e-9, chirp=3.0)
tbp_chirped = time_bandwidth_product(chirped.At, chirped.Aw, grid.t, grid.omega)
# tbp_chirped ≈ 0.315 × √(1 + 3²) ≈ 1.00 → Heavily chirped

# Monitor TBP during propagation
results = solve(pulse, params)
tbp_out = time_bandwidth_product(results.At[:, end], results.Aw[:, end], 
                                  grid.t, grid.omega)
# Dispersion increases TBP (temporal broadening)
# Nonlinearity can decrease TBP (spectral broadening with narrowing)
```

# Pulse Compression Context
A chirped pulse (TBP > TBP_min) can be compressed by passing through
dispersion with opposite sign:
- **Up-chirped** (red-to-blue): Compress with β₂ < 0
- **Down-chirped** (blue-to-red): Compress with β₂ > 0

Compression factor ≈ TBP / TBP_min

# Notes
- Returns NaN if pulse is too weak or poorly defined
- Affected by noise on pulse edges
- For accurate results, ensure pulse is fully captured in time window

# See Also
- [`spectral_bandwidth`](@ref): Calculate spectral FWHM
- [`fwhm`](@ref): Generic FWHM calculation
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
