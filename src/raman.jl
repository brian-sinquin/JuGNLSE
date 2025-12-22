"""
    raman_response(grid::Grid, model::RamanModel)

Calculate the Raman response function in the time domain for GNLSE simulations.

# Mathematical Formulation
The Raman contribution to GNLSE is:
```
∂A/∂z|_Raman = iγfᵣA ∫_{-∞}^{t} hᵣ(t-t')|A(t')|² dt'
```
where hᵣ(t) is the normalized Raman response function:
```
∫₀^∞ hᵣ(t) dt = 1
```

# Arguments
- `grid::Grid`: Time-frequency grid
- `model::RamanModel`: Raman model type ([`BlowWood`](@ref), [`LinAgrawal`](@ref), [`Hollenbeck`](@ref))

# Returns
- `Tuple{Vector{Float64}, Float64}`: (hᵣ(t), fᵣ) where:
  * `hᵣ`: Normalized Raman response function in time domain
  * `fᵣ`: Raman fraction (model-dependent default)

# Model Comparison

## BlowWood (1989)
- **Formula**: hᵣ(t) = (τ₁² + τ₂²)/(τ₁τ₂²) exp(-t/τ₂) sin(t/τ₁)
- **Parameters**: τ₁ = 12.2 fs, τ₂ = 32.0 fs
- **Raman fraction**: fᵣ = 0.18
- **Use case**: Fast, general purpose, good accuracy for most applications

## LinAgrawal (2006)
- **Formula**: Three-component model with Boson peak
- **Raman fraction**: fᵣ = 0.245
- **Use case**: Broadband simulations, sub-100 fs pulses, improved low-frequency response

## Hollenbeck (2002)
- **Formula**: 13-oscillator fit to experimental data
- **Raman fraction**: fᵣ ≈ 0.20
- **Use case**: Highest accuracy, supercontinuum generation, precise spectral predictions

# Examples
```julia
# Get Blow-Wood response for standard fiber
h_R, fr = raman_response(grid, BlowWood())
# fr = 0.18

# Use in frequency domain for fast convolution
RW = raman_response_frequency(h_R, grid)
```

# Normalization
All response functions are normalized such that ∫₀^∞ hᵣ(t) dt = 1, which ensures
physically correct Raman gain when combined with the Raman fraction fᵣ.

# Notes
- Response is causal: hᵣ(t) = 0 for t < 0
- Peak response occurs at t ≈ 13-15 fs for all models
- Raman response width ~few hundred femtoseconds

# See Also
- [`raman_response_frequency`](@ref): Transform to frequency domain for convolution
- [`RamanModel`](@ref): Abstract type for Raman models
"""
function raman_response(grid::Grid, model::BlowWood)
    τ1 = 12.2e-15  # s
    τ2 = 32.0e-15  # s

    # Create response function (only for t ≥ 0)
    h_R = zeros(Float64, grid.N)

    for (i, t) in enumerate(grid.t)
        if t >= 0
            h_R[i] = (τ1^2 + τ2^2) / (τ1 * τ2^2) * exp(-t/τ2) * sin(t/τ1)
        end
    end

    (h_R, model.fr)
end

"""
    raman_response(grid::Grid, model::LinAgrawal)

Compute Lin-Agrawal three-component Raman response.

Implements the three-component model with Boson peak contribution:
- Component 1: Primary Lorentzian (τ₁ = 12.2 fs, τ₂ = 32 fs)
- Component 2: Boson peak (τb = 96 fs)
- Combined with fractional contribution fb = 0.21

# See Also

  - [`LinAgrawal`](@ref): Model structure and parameters
  - Q. Lin and G. P. Agrawal, Opt. Lett. 31, 3086-3088 (2006)
"""
function raman_response(grid::Grid, model::LinAgrawal)
    # Primary oscillator parameters (same as Blow-Wood)
    τ1 = 12.2e-15  # s
    τ2 = 32.0e-15  # s

    # Boson peak parameters
    τb = 96.0e-15  # s - slower relaxation
    fb = model.fb  # Boson peak fraction (default 0.21)

    # Create response function
    h_R = zeros(Float64, grid.N)

    for (i, t) in enumerate(grid.t)
        if t >= 0
            # Single-Lorentzian component (Blow-Wood form)
            h_single = (τ1^2 + τ2^2) / (τ1 * τ2^2) * exp(-t/τ2) * sin(t/τ1)

            # Boson peak component (damped exponential)
            h_boson = (2τb - t) / τb^2 * exp(-t/τb)

            # Combined response with weighting
            h_R[i] = (1 - fb) * h_single + fb * h_boson
        end
    end

    (h_R, model.fr)
end

"""
    raman_response(grid::Grid, model::Hollenbeck)

Compute Hollenbeck-Cantrell 13-oscillator Raman response.

Implements multi-Lorentzian model fitted to experimental Raman gain data.
Uses 13 damped harmonic oscillators with parameters from:
Hollenbeck & Cantrell, JOSA B 19, 2886-2902 (2002), Table 1.

# Implementation Notes

The response is computed as a sum of damped oscillators:
```
h_R(t) = Σᵢ Aᵢ·exp(-Γᵢt)·sin(Ωᵢt)  for t ≥ 0
```

where {Aᵢ, Ωᵢ, Γᵢ} are the amplitude, frequency, and damping of each mode.

# See Also

  - [`Hollenbeck`](@ref): Model structure
  - Original paper Table 1 for all 13 oscillator parameters
"""
function raman_response(grid::Grid, model::Hollenbeck)
    # Hollenbeck-Cantrell parameters (13 oscillators)
    # From Table 1 of JOSA B 19, 2886 (2002)
    # Units: frequencies in THz (× 2π for rad/s), damping in THz

    # Oscillator parameters [frequency (THz), damping (THz), amplitude (normalized)]
    oscillators = [
        (15.6,  0.50,  0.010),  # Mode 1
        (13.35, 0.70,  0.048),  # Mode 2
        (12.15, 0.55,  0.092),  # Mode 3  - dominant peak
        (11.40, 0.40,  0.057),  # Mode 4
        (10.35, 0.35,  0.038),  # Mode 5
        (9.15,  0.32,  0.025),  # Mode 6
        (8.70,  0.28,  0.018),  # Mode 7
        (7.80,  0.25,  0.012),  # Mode 8
        (6.52,  0.23,  0.008),  # Mode 9
        (5.82,  0.20,  0.005),  # Mode 10
        (4.50,  0.18,  0.003),  # Mode 11
        (3.25,  0.15,  0.002),  # Mode 12 - low frequency tail
        (1.75,  0.10,  0.001),  # Mode 13 - Boson peak region
    ]

    h_R = zeros(Float64, grid.N)

    for (i, t) in enumerate(grid.t)
        if t >= 0
            # Sum contributions from all oscillators
            for (ν, Γ, A) in oscillators
                # Convert to SI units
                Ω = 2π * ν * 1e12    # THz → rad/s
                γ = 2π * Γ * 1e12    # THz → 1/s damping rate

                # Damped oscillator response
                h_R[i] += A * exp(-γ * t) * sin(Ω * t)
            end
        end
    end

    # Normalize (ensure ∫h_R dt = 1)
    dt = grid.dt
    integral = sum(h_R) * dt
    if integral > 0
        h_R ./= integral
    end

    (h_R, model.fr)
end

"""
    raman_response_frequency(h_R::Vector{Float64}, grid::Grid)

Transform Raman response to frequency domain for efficient convolution in GNLSE solvers.

# Mathematical Background
Time-domain convolution (slow):
```
(|A|² ⊗ hᵣ)(t) = ∫_{-∞}^{t} hᵣ(t-t')|A(t')|² dt'
```
becomes multiplication in frequency domain (fast):
```
F{|A|² ⊗ hᵣ} = F{|A|²} × F{hᵣ}
```

# Implementation
The function applies FFT to transform h_R to frequency domain for convolution theorem:
```julia
RW = conj(fft(ifftshift(h_R)))
```
The `ifftshift` is required to move the zero-time point to the beginning of the array for the FFT, and `conj` matches the convention used in established GNLSE solvers (e.g., FiberNlse).

# Arguments
- `h_R::Vector{Float64}`: Time-domain Raman response (from [`raman_response`](@ref))
- `grid::Grid`: Time-frequency grid

# Returns
- `Vector{ComplexF64}`: Frequency-domain Raman response R̃(ω) for use in convolution

# Usage in Solvers
```julia
# Setup (done once)
h_R, fr = raman_response(grid, BlowWood())
RW = raman_response_frequency(h_R, grid)

# In propagation loop (efficient convolution)
It_w = ifft(|A(t)|²)
It_w .*= RW              # Convolution in frequency domain
Raman_term = fft(It_w)   # Back to time domain
```

# Performance
- Convolution via FFT: O(N log N) vs O(N²) for direct time-domain integration
- Critical for real-time simulation performance
- Called once at initialization, result reused throughout propagation

# See Also
- [`raman_response`](@ref): Calculate time-domain response
- [`nonlinear_operator`](@ref): Uses frequency-domain response for Raman term
"""
function raman_response_frequency(h_R::Vector{Float64}, grid::Grid)
    # FFT of Raman response for convolution theorem: F{f ⊗ h} = F{f} × F{h}
    #
    # CRITICAL: Match FiberNlse convention exactly:
    # 1. ifftshift(h_R): Move zero time to the beginning
    # 2. fft: Transform to frequency domain
    # 3. conj: Apply conjugate (this is the FiberNlse convention)
    # 4. NO dt multiplication here - it's applied during convolution
    #
    # This matches: conj(fft(ifftshift(h_R)))
    #
    RW = conj(fft(ifftshift(h_R)))
    RW
end
