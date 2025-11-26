"""
    Medium{T<:Real}

Structure containing nonlinear waveguide medium parameters for GNLSE simulations.

# Fields
- `length::T`: Propagation length [m]
- `gamma::Union{T,Vector{T}}`: Nonlinear coefficient [W⁻¹m⁻¹], scalar or frequency-dependent
- `betas::Vector{T}`: Dispersion coefficients [sⁿ/m] in SI units
- `alpha::Union{T,Vector{T}}`: Loss coefficient [dB/km], can be constant or frequency-dependent
- `lambda0::T`: Center wavelength [m]
- `scaling::Union{Nothing,Vector{T}}`: Pseudo-envelope scaling factor [Aeff(ω₀)/Aeff(ω)]^(1/4) for M-GNLSE

# Notes
When `gamma` is a vector (frequency-dependent), the solver uses the Lægsgaard (2007) 
pseudo-envelope method to avoid convolution in the time domain. The `scaling` factor 
should be provided as [Aeff(ω₀)/Aeff(ω)]^(1/4) for M-GNLSE simulations.

# References
J. Lægsgaard, "Mode profile dispersion in the generalized nonlinear Schrödinger equation,"
Opt. Express 15, 16110-16123 (2007).
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

"""
    RamanModel

Abstract type for Raman response models.
"""
abstract type RamanModel end

"""
    BlowWood <: RamanModel

Blow-Wood (1989) Raman response model with single Lorentzian.
Default Raman fraction: fr = 0.18
"""
struct BlowWood <: RamanModel end

"""
    LinAgrawal <: RamanModel

Lin-Agrawal (2006) Raman response model including Boson peak.
Default Raman fraction: fr = 0.245
"""
struct LinAgrawal <: RamanModel end

"""
    Hollenbeck <: RamanModel

Hollenbeck-Cantrell (2002) 13-component Raman response model.
Most accurate experimental fit. Default Raman fraction: fr ≈ 0.20
"""
struct Hollenbeck <: RamanModel end

"""
    Grid{T<:Real}

Structure containing time and frequency grid information.

# Fields
- `N::Int`: Number of grid points (preferably power of 2)
- `t::Vector{T}`: Time grid [s]
- `omega::Vector{T}`: Angular frequency grid [rad/s]
- `dt::T`: Time step [s]
- `domega::T`: Frequency step [rad/s]
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

Structure representing a pulse envelope in time and frequency domains.

# Fields
- `At::Vector{T}`: Time domain envelope
- `Aw::Vector{T}`: Frequency domain envelope
- `grid::Grid`: Associated grid
"""
mutable struct Pulse{T<:Complex}
    At::Vector{T}
    Aw::Vector{T}
    grid::Grid
end

"""
    SimParams

Structure containing simulation parameters for GNLSE.

# Fields
- `medium::Medium`: Nonlinear medium parameters
- `N::Int`: Number of grid points
- `n_saves::Int`: Number of output points along propagation
- `raman::Bool`: Include Raman effect
- `shock::Bool`: Include self-steepening (shock) effect
- `raman_model::RamanModel`: Raman response model to use
- `fr::Float64`: Raman fraction
- `reltol::Float64`: Relative tolerance for adaptive integrator
- `abstol::Float64`: Absolute tolerance for adaptive integrator
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
