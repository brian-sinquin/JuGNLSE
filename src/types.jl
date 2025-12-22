"""
    Medium{T<:Real}

Nonlinear waveguide medium parameters for GNLSE propagation.

# Fields

  - `length::T`: Propagation length [m]
  - `gamma::Union{T,Vector{T}}`: Nonlinear coefficient [W⁻¹m⁻¹]
  - `betas::Vector{T}`: Dispersion coefficients [sⁿ/m], `betas[1]=β₂`, `betas[2]=β₃`, etc.
  - `alpha::Union{T,Vector{T}}`: Loss coefficient [Nepers/m]
  - `lambda0::T`: Center wavelength [m]
  - `scaling::Union{Nothing,Vector{T}}`: M-GNLSE scaling [Aeff(ω₀)/Aeff(ω)]^(1/4)

# Notes

  - Vector gamma enables M-GNLSE with frequency-dependent effective area (Lægsgaard 2007).
  - `scaling` is required for vector gamma simulations.
  - Beta array excludes β₀ and β₁.
"""
struct Medium{T <: Real}
    length::T
    gamma::Union{T, Vector{T}}
    betas::Vector{T}
    alpha::Union{T, Vector{T}}
    lambda0::T
    scaling::Union{Nothing, Vector{T}}

    function Medium{T}(
        length_m::T,
        gamma::Union{T, Vector{T}},
        betas::Vector{T},
        alpha::Union{T, Vector{T}},
        lambda0::T,
        scaling::Union{Nothing, Vector{T}}=nothing,
    ) where {T <: Real}
        length_m > 0 || throw(ArgumentError("Propagation length must be positive"))

        # Validate gamma
        if gamma isa Number
            gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        else
            all(gamma .>= 0) ||
                throw(ArgumentError("All nonlinear coefficients must be non-negative"))
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
                throw(
                    ArgumentError(
                        "Scaling factor only applies to frequency-dependent gamma (vector), not scalar gamma",
                    ),
                )
            end
            Base.length(scaling) == Base.length(gamma) || throw(
                ArgumentError(
                    "Scaling factor length ($(Base.length(scaling))) must match gamma length ($(Base.length(gamma)))",
                ),
            )
        end

        new{T}(length_m, gamma, betas, alpha, lambda0, scaling)
    end
end

# Constructor for scalar gamma (backward compatibility)
Medium(
    length::T, gamma::T, betas::Vector{T}, alpha::Union{T, Vector{T}}, lambda0::T
) where {T <: Real} = Medium{T}(length, gamma, betas, alpha, lambda0, nothing)

# Constructor for frequency-dependent gamma with scaling
Medium(
    length::T,
    gamma::Vector{T},
    betas::Vector{T},
    alpha::Union{T, Vector{T}},
    lambda0::T,
    scaling::Vector{T},
) where {T <: Real} = Medium{T}(length, gamma, betas, alpha, lambda0, scaling)

# Constructor for frequency-dependent gamma without scaling
Medium(
    length::T, gamma::Vector{T}, betas::Vector{T}, alpha::Union{T, Vector{T}}, lambda0::T
) where {T <: Real} = Medium{T}(length, gamma, betas, alpha, lambda0, nothing)

# Constructor that explicitly rejects scalar gamma with scaling (error case)
function Medium(
    length::T,
    gamma::T,
    betas::Vector{T},
    alpha::Union{T, Vector{T}},
    lambda0::T,
    scaling::Vector{T},
) where {T <: Real}
    throw(ArgumentError("Scaling factor only applies to frequency-dependent gamma"))
end

"""
    RamanModel

Abstract base type for Raman response models in GNLSE.
"""
abstract type RamanModel end

"""
    BlowWood <: RamanModel

Single Lorentzian Raman response model. Suitable for narrowband simulations with
pulse durations > 100 fs. Parameters: τ₁ = 12.2 fs, τ₂ = 32.0 fs, fr = 0.18.

Reference: K. J. Blow & D. Wood, IEEE J. Quantum Electron. 25, 2665 (1989).
"""
struct BlowWood <: RamanModel
    fr::Float64
    BlowWood(fr::Float64=0.18) = new(fr)
end

"""
    LinAgrawal <: RamanModel

Three-component Raman model with Boson peak. Provides improved accuracy for
broadband supercontinuum generation compared to single-Lorentzian model.

Parameters: τ₁ = 12.2 fs, τ₂ = 32 fs, τb = 96 fs, fb = 0.21, fr = 0.245.

Reference: Q. Lin & G. P. Agrawal, Opt. Lett. 31, 3086 (2006).
"""
struct LinAgrawal <: RamanModel
    fr::Float64
    fb::Float64
    LinAgrawal(; fr::Float64=0.245, fb::Float64=0.21) = new(fr, fb)
end

"""
    Hollenbeck <: RamanModel

13-oscillator Raman model fitted to measured Raman gain spectrum of fused silica.
Most accurate model for broadband simulations. Oscillators span 1.75-15.6 THz.

Parameters: 13 damped harmonic oscillators, fr = 0.20.

Reference: D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002).
"""
struct Hollenbeck <: RamanModel
    fr::Float64
    Hollenbeck(fr::Float64=0.20) = new(fr)
end

"""
    Grid{T<:Real}

Time and frequency grid for GNLSE propagation.

# Fields

  - `N::Int`: Number of grid points
  - `t::Vector{T}`: Time grid [s], centered at zero
  - `omega::Vector{T}`: Angular frequency detuning Δω = ω - ω₀ [rad/s]
  - `dt::T`: Time step [s]
  - `omega0::T`: Central angular frequency ω₀ [rad/s]
  - `lambda0::T`: Center wavelength λ₀ [m]
  - `lambda::Vector{T}`: Wavelength grid [m]

# Notes

  - `omega` is frequency detuning in FFT order (not monotonic)
  - Use `fft`/`ifft` directly without `fftshift`
"""
struct Grid{T <: Real}
    N::Int
    t::Vector{T}
    omega::Vector{T}
    dt::T
    omega0::T
    lambda0::T
    lambda::Vector{T}
end

"""
    Pulse{T<:Complex}

Optical pulse envelope in time and frequency domains.

# Fields

  - `At::Vector{T}`: Time domain envelope A(t) [√W]
  - `Aw::Vector{T}`: Frequency domain envelope A(ω) [√W·s]
  - `grid::Grid`: Associated time-frequency grid

# Notes

  - Related by FFT: `Aw = fft(At)`, `At = ifft(Aw)` (inverted FFT convention).
  - Energy: E = ∫|A(t)|²dt, Power: P(t) = |A(t)|².
"""
mutable struct Pulse{T <: Complex}
    At::Vector{T}
    Aw::Vector{T}
    grid::Grid
end

"""
    SimParams

Simulation parameters for GNLSE propagation.

# Fields

  - `medium::Medium`: Fiber medium parameters
  - `n_saves::Int`: Number of output snapshots
  - `raman::Bool`: Enable Raman scattering
  - `shock::Bool`: Enable self-steepening
  - `raman_model::RamanModel`: Raman response model
  - `fr::Float64`: Raman fraction (0 ≤ fr ≤ 1)
"""
struct SimParams
    medium::Medium
    n_saves::Int
    dz::Float64
    raman::Bool
    shock::Bool
    raman_model::RamanModel
    fr::Float64

    function SimParams(medium, n_saves, dz, raman, shock, raman_model, fr)
        n_saves > 0 || throw(ArgumentError("n_saves must be positive"))
        dz > 0 || throw(ArgumentError("Initial step size dz must be positive"))
        0 ≤ fr ≤ 1 || throw(ArgumentError("Raman fraction must be between 0 and 1"))
        new(medium, n_saves, dz, raman, shock, raman_model, fr)
    end
end

# Simplified constructor with keyword arguments
function SimParams(;
    medium::Medium,
    n_saves::Int=200,
    dz::Real=medium.length / (10 * n_saves), # Default conservative step
    raman::Bool=true,
    shock::Bool=true,
    raman_model::RamanModel=BlowWood(),
    fr::Float64=raman_model.fr,
    # Ignored parameters for backward compatibility
    N::Union{Int, Nothing}=nothing,
    reltol::Union{Float64, Nothing}=nothing,
    abstol::Union{Float64, Nothing}=nothing,
    nonlinearity_function::Union{Function, Nothing}=nothing,
)
    # Info/warn if deprecated parameters are used
    N !== nothing && @info "N parameter ignored - grid size taken from pulse"
    nonlinearity_function !== nothing &&
        @warn "nonlinearity_function ignored - auto-configured from physics flags"

    SimParams(medium, n_saves, Float64(dz), raman, shock, raman_model, fr)
end
