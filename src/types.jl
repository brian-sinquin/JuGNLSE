"""
Types and structures for JuGNLSE in natural SI units.

Adapted from: gnlse-python (https://github.com/WUST-FOG/gnlse-python)
"""

"""
    DispersionModel

Abstract base type for chromatic-dispersion models. A model maps the relative
angular-frequency grid `V = ω - ω₀` [rad/s] to the propagation-constant
deviation `B(V)` [1/m] used in the dispersion operator `D = iB - α/2`.

Concrete models: [`TaylorDispersion`](@ref), [`TabulatedDispersion`](@ref).
"""
abstract type DispersionModel end

"""
    TaylorDispersion(betas)

Dispersion from a Taylor expansion of the propagation constant about ω₀:

    B(V) = Σ βₙ / n! · Vⁿ ,   n ≥ 2

`betas[1] = β₂` [s²/m], `betas[2] = β₃` [s³/m], … (β₀ and β₁ are excluded).
"""
struct TaylorDispersion <: DispersionModel
    betas::Vector{Float64}
end

# An empty `betas` vector means no dispersion (pure SPM).
TaylorDispersion(betas::AbstractVector{<:Real}) = TaylorDispersion(collect(Float64, betas))

"""
    TabulatedDispersion(detuning, beta)

Dispersion from a measured/tabulated curve. `detuning` is the relative angular
frequency ω - ω₀ [rad/s] (sorted ascending); `beta` is the corresponding
propagation-constant deviation `B` [1/m] in the co-moving frame. Values are
linearly interpolated onto the simulation grid; outside the tabulated range the
nearest endpoint is held (flat extrapolation).
"""
struct TabulatedDispersion <: DispersionModel
    detuning::Vector{Float64}
    beta::Vector{Float64}

    function TabulatedDispersion(
        detuning::AbstractVector{<:Real}, beta::AbstractVector{<:Real}
    )
        length(detuning) == length(beta) ||
            throw(ArgumentError("detuning and beta must have equal length"))
        length(detuning) >= 2 ||
            throw(ArgumentError("need at least two tabulated samples"))
        issorted(detuning) ||
            throw(ArgumentError("detuning must be sorted ascending"))
        new(collect(Float64, detuning), collect(Float64, beta))
    end
end

"""
    Medium{T<:Real}

Fiber medium parameters for GNLSE propagation.

# Fields

  - `length::T`: Propagation length [m]
  - `gamma::T`: Nonlinear coefficient [1/(W·m)]
  - `loss::T`: Loss factor [dB/m]
  - `dispersion::DispersionModel`: Chromatic-dispersion model
  - `lambda0::T`: Center wavelength [m]

# Constructors

```julia
Medium(length, gamma, loss, betas::AbstractVector, lambda0)   # Taylor βₙ
Medium(length, gamma, loss, dispersion::DispersionModel, lambda0)
Medium(; length, gamma, loss=0.0, betas=…, dispersion=…, lambda0)
```

# Notes

  - Taylor `betas` exclude β₀ and β₁; `betas[1] = β₂`
  - loss in dB/m converted to α = ln(10^(loss/10)) Np/m in the operator
"""
struct Medium{T <: Real}
    length::T
    gamma::T
    loss::T
    dispersion::DispersionModel
    lambda0::T

    function Medium{T}(
        length::T, gamma::T, loss::T, dispersion::DispersionModel, lambda0::T
    ) where {T <: Real}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))

        new{T}(length, gamma, loss, dispersion, lambda0)
    end
end

# Positional constructor — Taylor betas (backward compatible)
Medium(
    length::T, gamma::T, loss::T, betas::AbstractVector{<:Real}, lambda0::T
) where {T <: Real} =
    Medium{T}(length, gamma, loss, TaylorDispersion(betas), lambda0)

# Positional constructor — explicit dispersion model
Medium(
    length::T, gamma::T, loss::T, dispersion::DispersionModel, lambda0::T
) where {T <: Real} =
    Medium{T}(length, gamma, loss, dispersion, lambda0)

"""
    Medium(; length, gamma, loss=0.0, betas=nothing, dispersion=nothing, lambda0)

Keyword constructor — avoids positional-argument mistakes. Supply the dispersion
either as Taylor `betas` or as an explicit `dispersion::DispersionModel` (exactly
one of the two).
"""
function Medium(;
    length::Real,
    gamma::Real,
    loss::Real=0.0,
    betas::Union{AbstractVector{<:Real}, Nothing}=nothing,
    dispersion::Union{DispersionModel, Nothing}=nothing,
    lambda0::Real,
)
    (betas === nothing) ⊻ (dispersion === nothing) ||
        throw(ArgumentError("provide exactly one of `betas` or `dispersion`"))
    disp = dispersion === nothing ? TaylorDispersion(betas) : dispersion
    T = promote_type(
        typeof(length), typeof(gamma), typeof(loss), typeof(lambda0), Float64
    )
    Medium{T}(T(length), T(gamma), T(loss), disp, T(lambda0))
end

"""
    Grid{T<:Real}

Time and frequency grid for GNLSE propagation.

# Fields

  - `N::Int`: Number of grid points (resolution)
  - `t::Vector{T}`: Time grid [s], centered at zero
  - `V::Vector{T}`: Relative angular frequency ω - ω₀ [rad/s], monotonic
  - `W::Vector{T}`: Absolute optical angular frequency ω = ω₀ + V [rad/s], monotonic
  - `dt::T`: Time step [s]
  - `omega0::T`: Central angular frequency ω₀ [rad/s]
  - `lambda0::T`: Center wavelength [m]

# Notes

  - V = ω - ω₀ is the physical detuning; W = ω₀ + V is the absolute frequency
  - ω₀ = 2πc/λ₀ (all frequencies in rad/s)
  - V and W are stored in monotonic order, not FFT-natural order
"""
struct Grid{T <: Real}
    N::Int
    t::Vector{T}
    V::Vector{T}          # Relative angular frequency ω - ω₀ [rad/s], monotonic
    W::Vector{T}          # Absolute angular frequency ω₀ + V [rad/s], monotonic
    dt::T
    omega0::T             # Central angular frequency [rad/s]
    lambda0::T            # Center wavelength [m]
end

"""
    RamanModel

Abstract base type for Raman response models.
"""
abstract type RamanModel end

"""
    AbstractGNLSESolver

Abstract type for GNLSE solvers.
"""
abstract type AbstractGNLSESolver end

"""
    AbstractGammaCoefficient

Abstract base type for nonlinear coefficient models.
"""
abstract type AbstractGammaCoefficient end

"""
    ConstantGamma(gamma)

Nonlinear coefficient `gamma` [1/(W·m)] assumed to be constant.
"""
struct ConstantGamma <: AbstractGammaCoefficient
    gamma::Float64
end

ConstantGamma(gamma::Real) = ConstantGamma(Float64(gamma))

"""
    ZDependentGamma(gamma_func::Function)

Nonlinear coefficient `gamma` [1/(W·m)] that varies with propagation distance `z` [m].
`gamma_func` is a function `f(z)` that returns the gamma value at a given `z`.
"""
struct ZDependentGamma <: AbstractGammaCoefficient
    gamma_func::Function
end

"""
    WavelengthDependentGamma(gamma_func::Function)

Nonlinear coefficient `gamma` [1/(W·m)] that varies with wavelength `lambda` [m].
`gamma_func` is a function `f(lambda)` that returns the gamma value at a given `lambda`.
"""
struct WavelengthDependentGamma <: AbstractGammaCoefficient
    gamma_func::Function
end

"""
    GNLSEProblem{T<:Complex}

Structure containing all parameters for a GNLSE simulation.
"""
struct GNLSEProblem{T <: Complex}
    medium::Medium
    grid::Grid
    initial_pulse::Pulse{T}
    sim_params::SimParams
    gamma_coefficient::AbstractGammaCoefficient
end

"""
    BlowWood <: RamanModel

Single Lorentzian Raman response model from K. J. Blow & D. Wood.

Parameters (SI units):

  - fr = 0.18: Raman fraction
  - τ₁ = 12.2 fs
  - τ₂ = 32 fs

Reference: K. J. Blow & D. Wood, IEEE J. Quantum Electron. 25, 2665 (1989)
"""
struct BlowWood <: RamanModel
    fr::Float64
    tau1::Float64
    tau2::Float64

    BlowWood(; fr::Float64=0.18, tau1::Float64=12.2e-15, tau2::Float64=32.0e-15) =
        new(fr, tau1, tau2)
end

"""
    LinAgrawal <: RamanModel

Three-component Raman model from Q. Lin & G. P. Agrawal.

Parameters (SI units):

  - fr = 0.245: Raman fraction
  - τ₁ = 12.2 fs
  - τ₂ = 32 fs
  - τb = 96 fs
  - fb = 0.21
  - fc = 0.04

Reference: Q. Lin & G. P. Agrawal, Opt. Lett. 31, 3086 (2006)
"""
struct LinAgrawal <: RamanModel
    fr::Float64
    tau1::Float64
    tau2::Float64
    taub::Float64
    fb::Float64
    fc::Float64

    LinAgrawal(;
        fr::Float64=0.245,
        tau1::Float64=12.2e-15,
        tau2::Float64=32.0e-15,
        taub::Float64=96.0e-15,
        fb::Float64=0.21,
        fc::Float64=0.04,
    ) = new(fr, tau1, tau2, taub, fb, fc)
end

"""
    Hollenbeck <: RamanModel

13-oscillator Raman model from D. Hollenbeck & C. D. Cantrell.

Parameters:

  - fr = 0.20: Raman fraction

Reference: D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002)
"""
struct Hollenbeck <: RamanModel
    fr::Float64

    Hollenbeck(; fr::Float64=0.20) = new(fr)
end



"""
    Pulse{T<:Complex}

Optical pulse envelope in time and frequency domains.

# Fields

  - `At::Vector{T}`: Time domain envelope A(t) [√W]
  - `AW::Vector{T}`: Frequency domain envelope A(ω) [√W·ps]
  - `grid::Grid`: Associated time-frequency grid

# Notes

Following gnlse-python convention:

  - AW = N * ifft(At) (note: inverted FFT convention)
  - At = fft(AW)
  - Power: P(t) = |A(t)|²
  - Energy: E = ∫|A(t)|²dt
"""
mutable struct Pulse{T <: Complex}
    At::Vector{T}
    AW::Vector{T}
    grid::Grid
end

"""
    SimParams

Simulation parameters for GNLSE propagation.

# Fields

  - `medium::Medium`: Fiber medium parameters
  - `z_saves::Int`: Number of snapshots to save along fiber
  - `raman_model::Union{RamanModel,Nothing}`: Raman scattering model (Nothing to disable)
  - `self_steepening::Bool`: Enable self-steepening/shock effect
  - `rtol::Float64`: Relative tolerance for adaptive ODE solver
  - `atol::Float64`: Absolute tolerance for adaptive ODE solver

# Notes

Following gnlse-python's GNLSESetup structure:

  - z_saves determines memory allocation for result arrays
  - Raman disabled by setting raman_model = nothing or fr = 0
  - Self-steepening modifies W frequency grid
"""
struct SimParams
    medium::Medium
    z_saves::Int
    raman_model::Union{RamanModel, Nothing}
    self_steepening::Bool
    rtol::Float64
    atol::Float64

    function SimParams(
        medium::Medium,
        z_saves::Int,
        raman_model::Union{RamanModel, Nothing},
        self_steepening::Bool,
        rtol::Float64,
        atol::Float64,
    )
        z_saves > 0 || throw(ArgumentError("z_saves must be positive"))
        rtol > 0 || throw(ArgumentError("rtol must be positive"))
        atol > 0 || throw(ArgumentError("atol must be positive"))

        new(medium, z_saves, raman_model, self_steepening, rtol, atol)
    end
end

# Simplified constructor with keyword arguments
function SimParams(;
    medium::Medium,
    z_saves::Int=200,
    raman_model::Union{RamanModel, Nothing}=BlowWood(),
    self_steepening::Bool=false,
    rtol::Float64=1e-6,
    atol::Float64=1e-8,
)
    SimParams(medium, z_saves, raman_model, self_steepening, rtol, atol)
end

"""
    Solution{T<:Complex}

Solution to GNLSE propagation.

# Fields

  - `t::Vector{Float64}`: Time domain grid [ps]
  - `W::Vector{Float64}`: Absolute angular frequency grid [rad/ps = THz]
  - `omega0::Float64`: Central angular frequency [rad/ps = THz]
  - `Z::Vector{Float64}`: Propagation distances [m]
  - `At::Matrix{T}`: Time domain solution (N × z_saves)
  - `AW::Matrix{T}`: Frequency domain solution (N × z_saves)

# Notes

Following gnlse-python's Solution structure:

  - At[i, j] = pulse at time t[i], distance Z[j]
  - AW[i, j] = frequency component at W[i], distance Z[j]
"""
struct Solution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Matrix{T}
    AW::Matrix{T}
end
