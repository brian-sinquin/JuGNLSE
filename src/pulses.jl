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
