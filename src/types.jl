"""
    Medium{T<:Real}

Structure containing nonlinear waveguide medium parameters for GNLSE simulations.

# Fields
- `length::T`: Propagation length [m]
- `gamma::Union{T,Vector{T}}`: Nonlinear coefficient [W⁻¹m⁻¹], scalar or frequency-dependent
- `betas::Vector{T}`: Dispersion coefficients βₙ [sⁿ/m] where `betas[1]=β₂`, `betas[2]=β₃`, etc.
- `alpha::Union{T,Vector{T}}`: Loss coefficient [dB/km], scalar or frequency-dependent
- `lambda0::T`: Center wavelength [m]
- `scaling::Union{Nothing,Vector{T}}`: Pseudo-envelope scaling [Aeff(ω₀)/Aeff(ω)]^(1/4) for M-GNLSE

# Examples
```julia
# Standard single-mode fiber (scalar gamma)
medium = Medium(
    0.15,                    # 15 cm fiber length
    0.11,                    # γ = 0.11 W⁻¹m⁻¹
    [-11.83e-27, 8.03e-41],  # β₂ = -11.83 ps²/m, β₃ = 8.03 ps³/m
    0.0,                     # lossless
    835e-9                   # λ₀ = 835 nm
)

# Frequency-dependent gamma with M-GNLSE (requires scaling)
gamma_vec = collect(range(0.10, 0.12, length=1024))
scaling = ones(1024)  # Compute from effective area: [Aeff(ω₀)/Aeff(ω)]^0.25
medium = Medium(0.15, gamma_vec, [-11.83e-27], 0.0, 835e-9, scaling)
```

# Notes
- **Scalar gamma**: Standard GNLSE with constant nonlinearity across spectrum
- **Vector gamma**: M-GNLSE using Lægsgaard (2007) pseudo-envelope method to handle frequency-dependent 
  effective area Aeff(ω). Requires `scaling` factor to avoid time-domain convolution.
- **Beta coefficients**: Use SI units! β₂ in s²/m, β₃ in s³/m, etc. Note: `betas[1]` is β₂, not β₀.
- **Loss**: Can be scalar (constant) or vector (frequency-dependent), specified in dB/km

# See Also
- [`create_grid`](@ref): Create time-frequency grid for simulation
- [`SimParams`](@ref): Complete simulation parameter structure

# References
- J. Lægsgaard, "Mode profile dispersion in the generalized nonlinear Schrödinger equation,"
  Opt. Express 15, 16110-16123 (2007). DOI: 10.1364/OE.15.016110
"""
struct Medium{T<:Real}
    length::T
    gamma::Union{T,Vector{T}}
    betas::Vector{T}
    alpha::Union{T,Vector{T}}
    lambda0::T
    scaling::Union{Nothing,Vector{T}}
    
    function Medium{T}(length_m::T, gamma::Union{T,Vector{T}}, betas::Vector{T}, 
                      alpha::Union{T,Vector{T}}, lambda0::T, 
                      scaling::Union{Nothing,Vector{T}}=nothing) where T<:Real
        length_m > 0 || throw(ArgumentError("Propagation length must be positive"))
        
        # Validate gamma
        if gamma isa Number
            gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        else
            all(gamma .>= 0) || throw(ArgumentError("All nonlinear coefficients must be non-negative"))
        end
        
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        
        # If gamma is vector, scaling should be provided
        if gamma isa Vector && scaling === nothing
            @warn "Frequency-dependent gamma provided without scaling factor. " *
                  "For accurate M-GNLSE simulations, provide scaling = [Aeff(ω₀)/Aeff(ω)]^(1/4)."
        end
        
        # Validate scaling if provided
        if scaling !== nothing
            if gamma isa Number
                throw(ArgumentError("Scaling factor only applies to frequency-dependent gamma"))
            end
            Base.length(scaling) == Base.length(gamma) || 
                throw(ArgumentError("Scaling factor must have same length as gamma"))
        end
        
        new{T}(length_m, gamma, betas, alpha, lambda0, scaling)
    end
end

# Constructor for scalar gamma (backward compatibility)
Medium(length::T, gamma::T, betas::Vector{T}, alpha::Union{T,Vector{T}}, lambda0::T) where T<:Real = 
    Medium{T}(length, gamma, betas, alpha, lambda0, nothing)

# Constructor for frequency-dependent gamma with scaling
Medium(length::T, gamma::Vector{T}, betas::Vector{T}, alpha::Union{T,Vector{T}}, lambda0::T, 
       scaling::Vector{T}) where T<:Real = 
    Medium{T}(length, gamma, betas, alpha, lambda0, scaling)

# Constructor for frequency-dependent gamma without scaling
Medium(length::T, gamma::Vector{T}, betas::Vector{T}, alpha::Union{T,Vector{T}}, lambda0::T) where T<:Real = 
    Medium{T}(length, gamma, betas, alpha, lambda0, nothing)

# Constructor that explicitly rejects scalar gamma with scaling (error case)
function Medium(length::T, gamma::T, betas::Vector{T}, alpha::Union{T,Vector{T}}, lambda0::T, 
                scaling::Vector{T}) where T<:Real
    throw(ArgumentError("Scaling factor only applies to frequency-dependent gamma"))
end

"""
    RamanModel

Abstract type for Raman response models used in GNLSE simulations.

# Available Models
- [`BlowWood`](@ref): Single Lorentzian, fr = 0.18
- [`LinAgrawal`](@ref): Three-component with Boson peak, fr = 0.245  
- [`Hollenbeck`](@ref): 13-oscillator experimental fit, fr ≈ 0.20

# See Also
- [`raman_response`](@ref): Calculate Raman response function
"""
abstract type RamanModel end

"""
    BlowWood <: RamanModel

Blow-Wood (1989) Raman response model with single Lorentzian oscillator.

# Parameters
- τ₁ = 12.2 fs (oscillation period)
- τ₂ = 32.0 fs (damping time)
- Default Raman fraction: fr = 0.18

# Notes
Computationally efficient model suitable for most simulations. Response function:
```
h_R(t) = (τ₁² + τ₂²)/(τ₁τ₂²) · exp(-t/τ₂) · sin(t/τ₁)  for t ≥ 0
```

# References
- K. J. Blow and D. Wood, "Theoretical description of transient stimulated Raman scattering in optical fibers,"
  IEEE J. Quantum Electron. 25, 2665-2673 (1989). DOI: 10.1109/3.40655
"""
struct BlowWood <: RamanModel end

"""
    LinAgrawal <: RamanModel

Lin-Agrawal (2006) three-component Raman response model including Boson peak contribution.

# Parameters
- Three components: main peak (75%), Boson peak (21%), secondary peak (4%)
- Default Raman fraction: fr = 0.245

# Notes
More accurate than Blow-Wood for broadband simulations. Captures low-frequency Boson peak
which affects sub-100 fs pulses and supercontinuum generation.

# References
- Q. Lin and G. P. Agrawal, "Raman response function for silica fibers,"
  Opt. Lett. 31, 3086-3088 (2006). DOI: 10.1364/OL.31.003086
"""
struct LinAgrawal <: RamanModel end

"""
    Hollenbeck <: RamanModel

Hollenbeck-Cantrell (2002) 13-component Raman response model.

# Parameters
- 13 Lorentzian oscillators fitted to experimental data
- Default Raman fraction: fr ≈ 0.20
- Most accurate experimental fit available

# Notes
Provides best agreement with experimental Raman gain spectrum. Recommended for
high-accuracy simulations and supercontinuum generation studies.

**Current Implementation**: Simplified to single Lorentzian. Full 13-oscillator
implementation planned for future release.

# References
- D. Hollenbeck and C. D. Cantrell, "Multiple-vibrational-mode model for fiber-optic Raman gain spectrum
  and response function," J. Opt. Soc. Am. B 19, 2886-2892 (2002). DOI: 10.1364/JOSAB.19.002886
"""
struct Hollenbeck <: RamanModel end

"""
    Grid{T<:Real}

Structure containing time and frequency grid information for GNLSE simulations.

# Fields
- `N::Int`: Number of grid points (should be power of 2 for FFT efficiency)
- `t::Vector{T}`: Time grid [s], centered at t=0
- `omega::Vector{T}`: Angular frequency grid [rad/s], frequency detuning Δω = ω - ω₀
- `dt::T`: Time step [s]
- `domega::T`: Frequency step [rad/s]

# Notes
- `omega` represents frequency detuning from center: Δω = ω - ω₀
- Grid is pre-arranged for FFT: use `fft` and `ifft` directly (not `fftshift`)
- Time grid is symmetric: `t = [-T/2, ..., -dt, 0, dt, ..., T/2-dt]`

# Example
```julia
grid = create_grid(2^12, 10e-12, 835e-9)  # 4096 points, 10 ps window, 835 nm
# grid.N = 4096
# grid.t spans -5 ps to +5 ps
# grid.dt = 2.44 fs
```

# See Also
- [`create_grid`](@ref): Constructor function for creating grids
"""
struct Grid{T<:Real}
    N::Int
    t::Vector{T}
    omega::Vector{T}
    dt::T
    domega::T
end

"""
    Pulse{T<:Complex}

Structure representing an optical pulse in both time and frequency domains.

# Fields
- `At::Vector{T}`: Time domain envelope A(t) [√W]
- `Aw::Vector{T}`: Frequency domain envelope A(ω) [√W·s]
- `grid::Grid`: Associated time-frequency grid

# Notes
- Envelopes are related by FFT: `Aw = ifft(At)` and `At = fft(Aw)`
- Normalization: pulse energy E = ∫|A(t)|²dt = Σ|A[i]|²·dt
- Physical power: P(t) = |A(t)|² [W]

# Example
```julia
grid = create_grid(2^12, 10e-12, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
energy = sum(abs2.(pulse.At)) * grid.dt  # Calculate energy [J]
```

# See Also
- [`sech_pulse`](@ref), [`gaussian_pulse`](@ref), [`custom_pulse`](@ref)
"""
mutable struct Pulse{T<:Complex}
    At::Vector{T}
    Aw::Vector{T}
    grid::Grid
end

"""
    SimParams

Simulation parameters structure for GNLSE propagation.

# Fields
- `medium::Medium`: Nonlinear medium parameters (fiber properties)
- `N::Int`: Number of grid points (should be power of 2)
- `n_saves::Int`: Number of output snapshots along propagation
- `raman::Bool`: Enable Raman scattering effect
- `shock::Bool`: Enable self-steepening (shock term)
- `raman_model::RamanModel`: Raman response model ([`BlowWood`](@ref), [`LinAgrawal`](@ref), [`Hollenbeck`](@ref))
- `fr::Float64`: Raman fraction (0 ≤ fr ≤ 1), typical value 0.18 for silica
- `reltol::Float64`: Relative tolerance for adaptive solvers (RK4IP, ERK4IP)
- `abstol::Float64`: Absolute tolerance for adaptive solvers

# Example
```julia
# Standard fiber with Raman and shock effects
params = SimParams(
    medium = medium,
    N = 2^12,        # 4096 points
    n_saves = 200,   # 200 snapshots
    raman = true,
    shock = true,
    raman_model = Hollenbeck(),  # Most accurate model
    fr = 0.18        # Silica fiber
)

# High-accuracy simulation with tight tolerances
params = SimParams(
    medium = medium,
    reltol = 1e-8,   # Very tight relative tolerance
    abstol = 1e-10   # Very tight absolute tolerance
)
```

# Notes
- **N**: Power of 2 strongly recommended for FFT efficiency (e.g., 2^10, 2^12, 2^14)
- **n_saves**: Controls output resolution; more saves = more memory but finer z-resolution
- **Raman models**: 
  * [`BlowWood`](@ref): Fast, single Lorentzian (fr ≈ 0.18)
  * [`LinAgrawal`](@ref): Includes Boson peak (fr ≈ 0.245)
  * [`Hollenbeck`](@ref): Most accurate, 13-component (fr ≈ 0.20)
- **Tolerances**: Tighter tolerances increase accuracy but reduce step size (slower computation)

# See Also
- [`solve`](@ref): Main solver interface
- [`Medium`](@ref): Medium parameter structure
"""
struct SimParams
    medium::Medium
    N::Int
    n_saves::Int
    raman::Bool
    shock::Bool
    raman_model::RamanModel
    fr::Float64
    reltol::Float64
    abstol::Float64
    
    function SimParams(medium, N, n_saves, raman, shock, raman_model, fr, reltol, abstol)
        N > 0 && ispow2(N) || @warn "N should be a power of 2 for optimal FFT performance"
        n_saves > 0 || throw(ArgumentError("n_saves must be positive"))
        0 ≤ fr ≤ 1 || throw(ArgumentError("Raman fraction must be between 0 and 1"))
        new(medium, N, n_saves, raman, shock, raman_model, fr, reltol, abstol)
    end
end

# Default constructor with sensible defaults
function SimParams(;
    medium::Medium,
    N::Int = 2^12,
    n_saves::Int = 200,
    raman::Bool = true,
    shock::Bool = true,
    raman_model::RamanModel = Hollenbeck(),
    fr::Float64 = 0.18,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-9
)
    SimParams(medium, N, n_saves, raman, shock, raman_model, fr, reltol, abstol)
end
