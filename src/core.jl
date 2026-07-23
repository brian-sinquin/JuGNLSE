"""
Types and structures for JuGNLSE in natural SI units.

Adapted from: gnlse-python (https://github.com/WUST-FOG/gnlse-python)
"""

"""
    DispersionModel

Abstract base type for chromatic-dispersion models.
"""
abstract type DispersionModel end

struct TaylorDispersion <: DispersionModel
    betas::Vector{Float64}
end
TaylorDispersion(betas::AbstractVector{<:Real}) = TaylorDispersion(collect(Float64, betas))

struct TabulatedDispersion <: DispersionModel
    detuning::Vector{Float64}
    beta::Vector{Float64}
    function TabulatedDispersion(detuning::AbstractVector{<:Real}, beta::AbstractVector{<:Real})
        length(detuning) == length(beta) || throw(ArgumentError("detuning and beta must have equal length"))
        length(detuning) >= 2 || throw(ArgumentError("need at least two tabulated samples"))
        issorted(detuning) || throw(ArgumentError("detuning must be sorted ascending"))
        new(collect(Float64, detuning), collect(Float64, beta))
    end
end

struct Medium{T <: Real}
    length::T
    gamma::T
    loss::T
    dispersion::DispersionModel
    lambda0::T

    function Medium{T}(length::T, gamma::T, loss::T, dispersion::DispersionModel, lambda0::T) where {T <: Real}
        length > 0 || throw(ArgumentError("Fiber length must be positive"))
        gamma >= 0 || throw(ArgumentError("Nonlinear coefficient must be non-negative"))
        loss >= 0 || throw(ArgumentError("Loss must be non-negative"))
        lambda0 > 0 || throw(ArgumentError("Center wavelength must be positive"))
        new{T}(length, gamma, loss, dispersion, lambda0)
    end
end

# Positional constructor — Taylor betas (backward compatible)
Medium(length::T, gamma::T, loss::T, betas::AbstractVector{<:Real}, lambda0::T) where {T <: Real} =
    Medium{T}(length, gamma, loss, TaylorDispersion(betas), lambda0)

# Positional constructor — explicit dispersion model
Medium(length::T, gamma::T, loss::T, dispersion::DispersionModel, lambda0::T) where {T <: Real} =
    Medium{T}(length, gamma, loss, dispersion, lambda0)

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
    T = promote_type(typeof(length), typeof(gamma), typeof(loss), typeof(lambda0), Float64)
    Medium{T}(T(length), T(gamma), T(loss), disp, T(lambda0))
end

struct Grid{T <: Real}
    N::Int
    t::Vector{T}
    V::Vector{T}
    W::Vector{T}
    dt::T
    omega0::T
    lambda0::T
end

abstract type RamanModel end

struct BlowWood <: RamanModel
    fr::Float64
    tau1::Float64
    tau2::Float64
    BlowWood(; fr::Float64=0.18, tau1::Float64=12.2e-15, tau2::Float64=32.0e-15) =
        new(fr, tau1, tau2)
end

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

struct Hollenbeck <: RamanModel
    fr::Float64
    Hollenbeck(; fr::Float64=0.20) = new(fr)
end

abstract type AbstractGNLSESolver end

abstract type AbstractGammaCoefficient end

struct ConstantGamma <: AbstractGammaCoefficient
    gamma::Float64
end
ConstantGamma(gamma::Real) = ConstantGamma(Float64(gamma))

struct ZDependentGamma <: AbstractGammaCoefficient
    gamma_func::Function
end

struct WavelengthDependentGamma <: AbstractGammaCoefficient
    gamma_func::Function
end

mutable struct Pulse{T <: Complex}
    At::Vector{T}
    AW::Vector{T}
    grid::Grid
end

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

struct GNLSEProblem{T <: Complex}
    medium::Medium
    grid::Grid
    initial_pulse::Pulse{T}
    sim_params::SimParams
    gamma_coefficient::AbstractGammaCoefficient
end

struct Solution{T <: Complex}
    t::Vector{Float64}
    W::Vector{Float64}
    omega0::Float64
    Z::Vector{Float64}
    At::Matrix{T}
    AW::Matrix{T}
end
"""
Grid generation for GNLSE simulations in natural SI units.

Units: Time in s, frequency in rad/s, wavelength in m
"""

"""
    create_grid(resolution::Int, time_window::Real, wavelength::Real)

Create time-frequency grid for GNLSE simulations in natural SI units.

# Arguments

  - `resolution::Int`: Number of grid points (power of 2 recommended)
  - `time_window::Real`: Total time window [s]
  - `wavelength::Real`: Center wavelength [m]

# Returns

  - `Grid`: Grid structure with:

      + t: time grid [s] spanning [-time_window/2, time_window/2]
      + V: relative angular frequency ω - ω₀ [rad/s], monotonic
      + W: absolute angular frequency ω = ω₀ + V [rad/s], monotonic
      + dt: time step [s]
      + omega0: central angular frequency ω₀ [rad/s]
      + lambda0: center wavelength [m]

# Notes

  - ω₀ = 2πc/λ₀ where c = 299792458 m/s, gives ω₀ in rad/s
  - V = 2π · [-N/2, ..., N/2-1] / (N·dt) [rad/s] is the relative frequency
    (physical detuning ω - ω₀), monotonic ordering
  - W = ω₀ + V [rad/s] is the absolute optical frequency

`V` and `W` are stored in monotonic (not FFT-natural) order; operators that act
on FFT output apply `ifftshift` as needed.
"""
function create_grid(resolution::Int, time_window::Real, wavelength::Real)
    resolution > 0 || throw(ArgumentError("resolution must be positive"))
    ispow2(resolution) ||
        @warn "resolution should be a power of 2 for optimal FFT performance"
    time_window > 0 || throw(ArgumentError("time_window must be positive"))
    wavelength > 0 || throw(ArgumentError("wavelength must be positive"))

    N = resolution

    # Time domain grid [s]
    t = collect(range(-time_window / 2, time_window / 2; length=N))
    dt = t[2] - t[1]

    # Relative angular frequency grid [rad/s], monotonic.
    # This is the physical detuning ω - ω₀: the package uses the standard optics
    # FFT convention (envelope spectrum AW = ifft(At), field At = fft(AW)), so a
    # spectral component at V evolves in time as exp(-iVt).
    V = 2π .* ((-N ÷ 2):(N ÷ 2 - 1)) ./ (N * dt)

    # Central angular frequency [rad/s]: ω₀ = 2πc/λ₀
    omega0 = (2.0 * π * c) / wavelength

    # Absolute optical angular frequency grid ω = ω₀ + V [rad/s]
    W = omega0 .+ V

    Grid{Float64}(N, t, V, W, dt, omega0, wavelength)
end

"""
    wavelength_grid(grid::Grid)
    wavelength_grid(solution::Solution)

Wavelength grid [m] for the absolute frequency axis, λ = 2πc/ω. The result is
aligned element-for-element with `grid.W` (and with `solution.W` / the columns
of `solution.AW`), so it is monotonically decreasing in array order.
"""
wavelength_grid(grid::Grid) = (2π * c) ./ grid.W
wavelength_grid(solution::Solution) = (2π * c) ./ solution.W
"""
Pulse envelope generation in natural SI units.

Units: Time in [s], Power in [W], Wavelength in [m]
"""

using FFTW
using Random

"""
    sech_pulse(grid::Grid, Pmax::Real, FWHM::Real)

Generate hyperbolic secant pulse in natural SI units.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: Peak power [W]
  - `FWHM::Real`: Pulse duration Full-Width Half-Maximum [s]

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

Following gnlse-python SechEnvelope:

```python
m = 2 * log(1 + sqrt(2))
A(T) = sqrt(Pmax) * 2 / (exp(m*T/FWHM) + exp(-m*T/FWHM))
     = sqrt(Pmax) * sech(m*T/FWHM)
```

Where m = 2*arcsinh(1) ≈ 1.763 is the factor relating FWHM to 1/e half-width.
"""
function sech_pulse(grid::Grid, Pmax::Real, FWHM::Real)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))

    # gnlse-python: m = 2 * np.log(1 + np.sqrt(2))
    m = 2 * log(1 + sqrt(2))

    # gnlse-python: A(T) = sqrt(Pmax) * 2 / (exp(m*T/FWHM) + exp(-m*T/FWHM))
    At = similar(grid.t, ComplexF64)
    @. At = sqrt(Pmax) * 2 / (exp(m * grid.t / FWHM) + exp(-m * grid.t / FWHM))

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

"""
    gaussian_pulse(grid::Grid, Pmax::Real, FWHM::Real)

Generate Gaussian pulse following gnlse-python GaussianEnvelope.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: Peak power [W]
  - `FWHM::Real`: Pulse duration Full-Width Half-Maximum [s]

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

Following gnlse-python GaussianEnvelope, where `m = 4*log(2)` relates the 1/e²
half-width to the FWHM:

```python
A(T) = sqrt(Pmax) * exp(-m * 0.5 * T² / FWHM²)
```

This defines a pulse whose intensity drops to half-maximum at ±FWHM/2.
"""
function gaussian_pulse(grid::Grid, Pmax::Real, FWHM::Real)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))

    # gnlse-python: m = 4 * np.log(2)
    m = 4 * log(2)

    # gnlse-python: A(T) = sqrt(Pmax) * exp(-m * .5 * T**2 / FWHM**2)
    At = similar(grid.t, ComplexF64)
    @. At = sqrt(Pmax) * exp(-m * 0.5 * grid.t^2 / FWHM^2)

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

"""
    lorentzian_pulse(grid::Grid, Pmax::Real, FWHM::Real)

Generate Lorentzian pulse following gnlse-python LorentzianEnvelope.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: Peak power [W]
  - `FWHM::Real`: Pulse duration Full-Width Half-Maximum [s]

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

Following gnlse-python LorentzianEnvelope:

```python
m = 2 * sqrt(sqrt(2) - 1)
A(T) = sqrt(Pmax) / (1 + (m*T/FWHM)^2)
```
"""
function lorentzian_pulse(grid::Grid, Pmax::Real, FWHM::Real)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    FWHM > 0 || throw(ArgumentError("FWHM must be positive"))

    # gnlse-python: m = 2 * sqrt(sqrt(2) - 1)
    m = 2 * sqrt(sqrt(2) - 1)

    # gnlse-python: A(T) = sqrt(Pmax) / (1 + (m*T/FWHM)**2)
    At = similar(grid.t, ComplexF64)
    @. At = sqrt(Pmax) / (1 + (m * grid.t / FWHM)^2)

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end

"""
    cw_pulse(grid::Grid, Pmax::Real; Pn::Real=0.0, rng=Random.default_rng())

Generate a continuous-wave (CW) field with optional broadband temporal noise.

# Arguments

  - `grid::Grid`: Time-frequency grid
  - `Pmax::Real`: CW power [W]
  - `Pn::Real`: Power of the additive temporal noise floor [W] (default: 0.0)
  - `rng`: random source for the noise realization

# Returns

  - `Pulse`: Pulse structure with At and AW

# Physics

A constant-amplitude field `√Pmax` with, if `Pn > 0`, an additive seed of
amplitude `√Pn` and an *independent* uniformly random phase in every time bin:

    A(t) = √Pmax + √Pn · exp(i·2π·U(t)),   U(t) ~ Uniform[0, 1)

For a physically grounded quantum (one-photon-per-mode) or RIN seed on top of a
clean field, use [`add_noise`](@ref) instead.
"""
function cw_pulse(
    grid::Grid, Pmax::Real; Pn::Real=0.0, rng::Random.AbstractRNG=Random.default_rng()
)
    Pmax >= 0 || throw(ArgumentError("Peak power must be non-negative"))
    Pn >= 0 || throw(ArgumentError("Noise power must be non-negative"))

    N = grid.N

    # Constant-amplitude CW field in the time domain
    At = fill(ComplexF64(sqrt(Pmax)), N)

    # Add noise if requested — a fresh, independent random phase per time bin
    if Pn > 0
        At .+= sqrt(Pn) .* cis.(2π .* rand(rng, N))
    end

    # Envelope spectrum (standard optics convention: AW = ifft(At))
    AW = ifft(At)

    return Pulse(At, AW, grid)
end
"""
Dispersion operator for GNLSE simulations in natural SI units.
"""

"""
    propagation_constant(V, model::DispersionModel)

Propagation-constant deviation `B(V)` [1/m] for the dispersion `model`, sampled
on the relative angular-frequency grid `V = ω - ω₀` [rad/s]. This is an
intermediate quantity (intermediate in the frequency domain) used internally to
construct the dispersion operator [`dispersion_operator`](@ref).

# Method Implementations

For **TaylorDispersion**, computes the power-series expansion:

    B(V) = Σ βₙ/n! · Vⁿ,  n ≥ 2

This representation is fast and suits analytical studies, but assumes dispersion
is smooth and well-approximated by the first few terms.

For **TabulatedDispersion**, linearly interpolates the measured/numerically-computed
dispersion curve onto the simulation grid, then uses constant extrapolation beyond
the tabulated frequency range. This is more accurate for complex materials (PCF,
highly dispersive windows) but requires tabulated data.
"""
function propagation_constant(V::AbstractVector{Float64}, model::TaylorDispersion)
    # Taylor series: B = Σ βₙ/n! · Vⁿ, n ≥ 2
    B = zeros(Float64, length(V))
    for (i, beta) in enumerate(model.betas)
        n = i + 1  # betas[1]=β₂ → n=2, betas[2]=β₃ → n=3, etc.
        B .+= beta ./ factorial(n) .* (V .^ n)
    end
    return B
end

function propagation_constant(V::AbstractVector{Float64}, model::TabulatedDispersion)
    # Linear interpolation onto V, flat extrapolation outside the tabulated range
    xs, ys = model.detuning, model.beta
    B = similar(V)
    @inbounds for k in eachindex(V, B)
        x = V[k]
        if x <= xs[1]
            B[k] = ys[1]
        elseif x >= xs[end]
            B[k] = ys[end]
        else
            j = searchsortedlast(xs, x)         # xs[j] ≤ x < xs[j+1]
            t = (x - xs[j]) / (xs[j + 1] - xs[j])
            B[k] = ys[j] + t * (ys[j + 1] - ys[j])
        end
    end
    return B
end

"""
    dispersion_operator(V::AbstractVector{Float64}, medium::Medium)

Construct the linear dispersion operator `D(V) = i·B(V) - α/2` [1/m], where:

  - `B(V)`: propagation-constant deviation from the dispersion model [1/m]
  - `α`: fiber loss in Neper/m, converted from dB/m via α = ln(10^(loss/10))
  - The factor i·B appears in the interaction-picture GNLSE; α/2 implements
    exponential decay

# Arguments

  - `V::Vector{Float64}`: relative angular frequency [rad/s]
  - `medium::Medium`: fiber with dispersion model and loss

# Returns

  - `D::Vector{ComplexF64}`: dispersion operator, one value per frequency bin

# Notes

The loss term `-α/2` in the frequency domain translates to multiplicative decay
`exp(-αz)` in the time domain (amplitude), which becomes `exp(-2αz)` in intensity.
See [`medium.loss`](@ref Medium) for units.
"""
function dispersion_operator(V::AbstractVector{Float64}, medium::Medium)
    alpha = log(10.0^(medium.loss / 10.0))
    B = propagation_constant(V, medium.dispersion)
    return @. 1im * B - alpha / 2
end

"""
    dispersion_operator(grid::Grid, medium::Medium)

Convenience wrapper that extracts `V` from `grid`.
"""
dispersion_operator(grid::Grid, medium::Medium) = dispersion_operator(grid.V, medium)
"""
Raman response functions following gnlse-python conventions.

Reference: gnlse-python raman_response.py
Units: Time in [s] (natural SI units)
"""

"""
    raman_response(T::Vector{Float64}, model::BlowWood)

Compute Raman response following gnlse-python raman_blowwood.

# Arguments
- `T::Vector{Float64}`: Time vector [ps]
- `model::BlowWood`: Raman model with parameters

# Returns
- `(fr, RT)`: Raman fraction and response function

# Physics
Following gnlse-python raman_blowwood:
```python
tau1 = 0.0122  # ps
tau2 = 0.032   # ps
ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
RT = ha
RT[T < 0] = 0
fr = 0.18
```

Reference: K. J. Blow & D. Wood, IEEE J. Quantum Electron. 25, 2665 (1989)
"""
function raman_response(T::Vector{Float64}, model::BlowWood)
    tau1 = model.tau1  # s
    tau2 = model.tau2  # s

    # gnlse-python: ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
    RT = (tau1^2 + tau2^2) / tau1 / (tau2^2) .* exp.(-T ./ tau2) .* sin.(T ./ tau1)

    # Apply causality
    # gnlse-python: RT[T < 0] = 0
    RT[T .< 0] .= 0

    return model.fr, RT
end

"""
    raman_response(T::Vector{Float64}, model::LinAgrawal)

Compute Raman response following gnlse-python raman_linagrawal.

# Arguments
- `T::Vector{Float64}`: Time vector [ps]
- `model::LinAgrawal`: Raman model with parameters

# Returns
- `(fr, RT)`: Raman fraction and response function

# Physics
Following gnlse-python raman_linagrawal:
```python
tau1 = 0.0122  # ps
tau2 = 0.032   # ps
taub = 0.096   # ps
fb = 0.21
fc = 0.04
fa = 1 - fb - fc
# Anisotropic response
ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
# Isotropic response
hb = (2*taub - T) / (taub**2) * exp(-T/taub)
# Total response
RT = (fa + fc) * ha + fb * hb
RT[T < 0] = 0
fr = 0.245
```

Reference: Q. Lin & G. P. Agrawal, Opt. Lett. 31, 3086 (2006)
"""
function raman_response(T::Vector{Float64}, model::LinAgrawal)
    tau1 = model.tau1  # s
    tau2 = model.tau2  # s
    taub = model.taub  # s
    fb = model.fb
    fc = model.fc
    fa = 1 - fb - fc

    # gnlse-python: ha = (tau1**2 + tau2**2) / tau1 / (tau2**2) * exp(-T/tau2) * sin(T/tau1)
    ha = (tau1^2 + tau2^2) / tau1 / (tau2^2) .* exp.(-T ./ tau2) .* sin.(T ./ tau1)

    # gnlse-python: hb = (2*taub - T) / (taub**2) * exp(-T/taub)
    hb = (2 .* taub .- T) ./ (taub^2) .* exp.(-T ./ taub)

    # gnlse-python: RT = (fa + fc) * ha + fb * hb
    RT = (fa + fc) .* ha .+ fb .* hb

    # Apply causality
    # gnlse-python: RT[T < 0] = 0
    RT[T .< 0] .= 0

    return model.fr, RT
end

"""
    raman_response(T::Vector{Float64}, model::Hollenbeck)

Compute Raman response following D. Hollenbeck & C. D. Cantrell's 13-oscillator fit.

# Arguments
  - `T::Vector{Float64}`: Time vector [s]
  - `model::Hollenbeck`: Raman model with Raman fraction fr

# Returns
  - `(fr, RT)`: Raman fraction `fr` and impulse response `RT(t)` [1/s]

# Physics

The Hollenbeck model combines 13 Lorentzian resonances with Gaussian spectral
broadening to fit experimental Raman gain/loss data from silica fiber:

    h(ω) = Σ Aᵢ [Lorentzian(ω - ωᵢ, Γᵢ) ⊗ Gaussian(ΔGᵢ)]

Each resonance is parametrized by:
  - **CP**: center position [cm⁻¹]
  - **A**: peak amplitude (relative units)
  - **Gauss**: Gaussian FWHM [cm⁻¹]
  - **Lorentz**: Lorentzian FWHM [cm⁻¹]

The model is converted to the time domain and normalized by the Raman fraction
`fr` (set to 0.20 by default), which represents the fractional power transfer
into the Raman-shifted component.

Reference: D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002)
"""
function raman_response(T::Vector{Float64}, model::Hollenbeck)
    # Component positions [1/cm]
    CP = [56.25, 100.0, 231.25, 362.5, 463.0, 497.0, 611.5, 691.67, 793.67,
          835.5, 930.0, 1080.0, 1215.0]

    # Peak intensity (amplitude)
    A = [1.0, 11.40, 36.67, 67.67, 74.0, 4.5, 6.8, 4.6, 4.2, 4.5, 2.7, 3.1, 3.0]

    # Gaussian FWHM [1/cm]
    Gauss = [52.10, 110.42, 175.00, 162.50, 135.33, 24.5, 41.5, 155.00, 59.5, 64.3,
             150.0, 91.0, 160.0]

    # Lorentzian FWHM [1/cm]
    Lorentz = [17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83, 51.67, 19.83, 21.43,
               50.00, 30.33, 53.33]

    # Convert wavenumbers [1/cm] to angular frequencies/rates [rad/s].
    # ω = 2π·c·(CP·100), with c in m/s; L and γ use π·c (FWHM convention).
    w = 2π .* c .* 100.0 .* CP
    L = π .* c .* 100.0 .* Gauss
    gamma = π .* c .* 100.0 .* Lorentz

    # Initialize RT
    RT = zeros(Float64, length(T))

    # gnlse-python: RT += A[i] * exp(-gamma[i]*T) * exp((-L[i]**2*T**2)/4) * sin(w[i]*T)
    for i in 1:length(A)
        @. RT += A[i] * exp(-gamma[i] * T) * exp((-L[i]^2 * T^2) / 4) * sin(w[i] * T)
    end

    # Apply causality
    # gnlse-python: RT[T < 0] = 0
    RT[T .< 0] .= 0

    # Normalize
    # gnlse-python: dt = T[1] - T[0]; RT = RT / (sum(RT) * dt)
    dt = T[2] - T[1]
    RT = RT ./ (sum(RT) * dt)

    return model.fr, RT
end

"""
    raman_response(grid::Grid, model::RamanModel)

Convenience wrapper that extracts time vector from grid.

# Arguments
- `grid::Grid`: Grid with time vector T
- `model::RamanModel`: Raman model

# Returns
- `(fr, RT)`: Raman fraction and response function
"""
function raman_response(grid::Grid, model::RamanModel)
    return raman_response(grid.t, model)
end

"""
    gamma(gamma_coeff::ConstantGamma, lambda::Real, z::Real)

Returns the constant nonlinear coefficient from a `ConstantGamma` model.
"""
function gamma(gamma_coeff::ConstantGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma
end

"""
    gamma(gamma_coeff::JuGNLSE.ZDependentGamma, lambda::Real, z::Real)

Returns the nonlinear coefficient from a `ZDependentGamma` model at a given `z`.
"""
function gamma(gamma_coeff::ZDependentGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma_func(z)
end

"""
    gamma(gamma_coeff::JuGNLSE.WavelengthDependentGamma, lambda::Real, z::Real)

Returns the nonlinear coefficient from a `WavelengthDependentGamma` model at a given `lambda`.
"""
function gamma(gamma_coeff::WavelengthDependentGamma, lambda::Real, z::Real)
    return gamma_coeff.gamma_func(lambda)
end
"""
Nonlinear operators for GNLSE propagation.

The package uses the standard optics FFT convention: the envelope spectrum is
`AW = ifft(At)` and the field is `At = fft(AW)`. So `to_freq` is `ifft` and
`to_time` is `fft`.
"""

using FFTW

# Conversion constant
const C = 299792458.0 # Speed of light in vacuum [m/s]

"""
    _omega_to_lambda(omega::Real)

Converts angular frequency `omega` [rad/s] to wavelength `lambda` [m].
"""
_omega_to_lambda(omega::Real) = 2π * C / omega

"""
    PhysicsModel

Pre-computed operators and FFT plans for GNLSE propagation.

# Fields
- `to_freq`: Plan for the time → frequency transform (`ifft`)
- `to_time`: Plan for the frequency → time transform (`fft`)
- `D`: Dispersion operator [1/m], FFT-natural order
- `gamma`: Nonlinear coefficient γ/ω₀ [s/(W·m·rad)]
- `W`: Nonlinear-term frequency factor [rad/s], FFT-natural order. Equals the
   absolute angular frequency ω₀+V when self-steepening is on, or the constant
   ω₀ when off — so the nonlinear term iγ·W reduces to iγ_phys in that case.
- `dt`: Time step [s]
- `N`: Number of grid points
- `fr`: Raman fraction
- `RW`: Raman response in frequency domain (if enabled)
- `buf_t1`, `buf_t2`: Pre-allocated time-domain buffers
- `buf_f1`: Pre-allocated frequency-domain buffer
"""
struct PhysicsModel{TF, TT, NL, GC <: AbstractGammaCoefficient}
    to_freq::TF
    to_time::TT
    D::Vector{ComplexF64}
    gamma_coefficient::GC  # Changed from gamma::Float64
    W::Vector{Float64}
    dt::Float64
    N::Int
    fr::Float64
    RW::Union{Nothing, Vector{ComplexF64}}
    nonlinear_function::NL
    lambda0::Float64 # Add lambda0 here
    # Pre-allocated buffers
    buf_t1::Vector{ComplexF64}
    buf_t2::Vector{ComplexF64}
    buf_f1::Vector{ComplexF64}
end

"""
    _spm(u, model::PhysicsModel)

Kerr (self-phase modulation) nonlinear operator.

Computes `iγ·W·to_freq(u|u|²)`. Self-steepening is carried by `model.W`: with
self-steepening off, `W = ω₀` (constant) and iγ·W = iγ_phys; with it on,
`W = ω₀+V`, giving the shock term iγ_phys(1+V/ω₀). Zero allocations.

# See also

[`_spm_raman`](@ref)
"""
function _spm(u, model::PhysicsModel, z::Float64)
    # SPM term u·|u|²
    @. model.buf_t1 = u * abs2(u)

    # Transform to the frequency domain
    mul!(model.buf_f1, model.to_freq, model.buf_t1)

    # Get gamma value using the central wavelength (for WavelengthDependentGamma simplification)
    # and the current z for ZDependentGamma
    gamma_val = gamma(model.gamma_coefficient, model.lambda0, z)

    # Multiply by iγW. The W in model.W already includes omega0 or omega0+V,
    # so we just need the physical gamma.
    @. model.buf_f1 = 1.0im * gamma_val * model.W * model.buf_f1

    return model.buf_f1
end

"""
    _spm_raman(u, model::PhysicsModel, z::Float64)

SPM with Raman scattering nonlinear operator.

Includes both instantaneous Kerr response and delayed Raman response via
convolution with hᵣ(t). Raman causes intrapulse frequency shift (red-shifting
spectral peak) and energy transfer to Stokes wavelengths. Zero allocations via
pre-allocated buffers.

# Physics

Total response: n₂[|E|² + fᵣ∫hᵣ(t-t')|E(t')|²dt']E where fᵣ ≈ 0.18 is the
Raman fraction. Convolution implemented efficiently via FFT multiplication.

# Implementation:

    conv = (|u|²) ⊛ hᵣ      via  to_time(to_freq(|u|²) .* RW)
    op   = u .* ((1-fr)|u|² + fr·dt·conv)
    result = iγ·W · to_freq(op)

# See also

[`raman_response`](@ref), [`_spm`](@ref)
"""
function _spm_raman(u, model::PhysicsModel, z::Float64)
    # `_spm_raman` is only selected when Raman is enabled, so RW is a Vector.
    # The assertion narrows the Union{Nothing,Vector} field type, keeping the
    # broadcast below type-stable and allocation-free.
    RW = model.RW::Vector{ComplexF64}

    # Intensity |u|²
    @. model.buf_t1 = abs2(u)

    # Raman convolution: multiply the intensity spectrum by the Raman response
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    @. model.buf_f1 = model.buf_f1 * RW
    mul!(model.buf_t2, model.to_time, model.buf_f1)   # buf_t2 = |u|² ⊛ h_R

    # Get gamma value for the current z and central lambda
    gamma_val = gamma(model.gamma_coefficient, model.lambda0, z)

    # Total nonlinearity (instantaneous Kerr + delayed Raman) times u
    @. model.buf_t1 = u * ((1.0 - model.fr) * abs2(u) + model.fr * model.dt * model.buf_t2)

    # Transform to the frequency domain and multiply by iγW
    mul!(model.buf_f1, model.to_freq, model.buf_t1)
    @. model.buf_f1 = 1.0im * gamma_val * model.W * model.buf_f1

    return model.buf_f1
end

"""
    choose_nonlinear_term(raman::Bool)

Select the nonlinear operator. Self-steepening is not a separate operator —
it is carried by `model.W` (see [`build_physics_model`](@ref)), so only the
presence of Raman scattering selects between operators.

# See also

[`build_physics_model`](@ref), [`_spm`](@ref), [`_spm_raman`](@ref)
"""
choose_nonlinear_term(raman::Bool) = raman ? _spm_raman : _spm

"""
    build_physics_model(grid::Grid, params::SimParams)

Construct PhysicsModel with pre-computed operators for GNLSE propagation.

Pre-computes all frequency-domain operators, FFT plans, and selects the
appropriate nonlinear function. Called once at start of solve() to enable
zero-allocation propagation in the ERK4IP stepper.

# Arguments

  - `grid`: Time-frequency grid
  - `params`: Simulation parameters (medium, physics flags)

# Returns

PhysicsModel struct ready for propagation

# Implementation Details

  - FFT plans use FFTW with FFTW.MEASURE flag for optimization
  - Dispersion operator computed via `dispersion_operator(grid, medium)`
  - Raman response computed in time domain then FFT'd to frequency domain
  - Self-steepening: folded into `W` (ω₀+Δω if enabled, else constant ω₀)
  - Nonlinear function selected via `choose_nonlinear_term(raman)`

# See also

[`PhysicsModel`](@ref), [`propagate_erk4ip`](@ref), [`dispersion_operator`](@ref),
[`raman_response`](@ref)
"""
function build_physics_model(grid::Grid, params::SimParams, gamma_coefficient::AbstractGammaCoefficient)
    medium = params.medium
    N = grid.N

    # Extract physics flags
    enable_raman = params.raman_model !== nothing
    enable_shock = params.self_steepening

    # Transform plans (FFTW.MEASURE for optimal performance). Standard optics
    # convention: time → frequency is ifft, frequency → time is fft.
    tmp = zeros(ComplexF64, N)
    to_freq = plan_ifft(tmp; flags=FFTW.MEASURE)
    to_time = plan_fft(tmp; flags=FFTW.MEASURE)

    # Compute dispersion operator. grid.V is in monotonic order, so the
    # operator must be fftshifted to FFT-natural order to align with AW.
    D = fftshift(dispersion_operator(grid, medium))

    # Gamma (nonlinear coefficient) - now stored as AbstractGammaCoefficient

    # Raman response in frequency domain (if enabled)
    raman_freq_response = nothing
    fr = 0.0
    if enable_raman
        # Compute the Raman response h_R(t) in the time domain
        fr, h_R = raman_response(grid, params.raman_model)

        # Frequency-domain Raman response. ifftshift puts the causal response
        # (zero for t<0) into FFT-natural order with zero delay at index 1.
        # RW = N·ifft(h_R) so that to_time(to_freq(I) .* RW) evaluates the
        # circular convolution I ⊛ h_R for the ifft/fft transform pair.
        raman_freq_response = N .* ifft(ifftshift(h_R))
    end

    # Frequency factor for the nonlinear term iγ·W·FFT(...).
    # With self-steepening: W = ω₀+Δω (the true absolute frequency), giving the
    # shock term iγ_phys(1+Δω/ω₀). Without it: W = ω₀ (constant), so iγ·W reduces
    # to iγ_phys. grid.W is monotonic; ifftshift puts it in FFT-natural order.
    W = enable_shock ? ifftshift(grid.W) : fill(grid.omega0, N)

    # Select nonlinear function (self-steepening is carried by W, not the operator)
    nonlinear_function = choose_nonlinear_term(enable_raman)

    # Pre-allocate working buffers for zero-allocation nonlinear operators
    buf_t1 = zeros(ComplexF64, N)
    buf_t2 = zeros(ComplexF64, N)
    buf_f1 = zeros(ComplexF64, N)

    # Construct model
    PhysicsModel(
        to_freq,
        to_time,
        D,
        gamma_coefficient, # Store the gamma_coefficient directly
        W,
        grid.dt,
        N,
        fr,
        raman_freq_response,
        nonlinear_function,
        grid.lambda0, # Pass lambda0
        buf_t1,
        buf_t2,
        buf_f1,
    )
end
