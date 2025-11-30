"""
    JuGNLSE

A high-performance Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE)  
for modeling nonlinear pulse propagation in optical fibers and waveguides.

# Overview

JuGNLSE provides state-of-the-art numerical methods for simulating complex nonlinear optical phenomena:
- **Soliton dynamics**: Fundamental and higher-order solitons, soliton fission
- **Supercontinuum generation**: Broadband spectral generation in PCF and HNLF
- **Modulation instability**: CW and quasi-CW evolution
- **Pulse compression/expansion**: Chirped pulse dynamics in dispersive media
- **Four-wave mixing**: Parametric processes and frequency conversion

# Key Features

## Physical Effects
✅ **Dispersion**: Arbitrary-order Taylor expansion (β₂, β₃, β₄, ...)  
✅ **Kerr nonlinearity**: Self-phase modulation (SPM), cross-phase modulation  
✅ **Raman scattering**: Three models (Blow-Wood, Lin-Agrawal, Hollenbeck)  
✅ **Self-steepening**: Shock formation for sub-100 fs pulses  
✅ **Fiber loss**: Constant or frequency-dependent α(ω)  
✅ **M-GNLSE**: Frequency-dependent γ(ω) via Lægsgaard pseudo-envelope method

## Numerical Methods
🚀 **RK4IP**: 4th-order Runge-Kutta in Interaction Picture (highest accuracy)  
🚀 **ERK4IP**: Embedded RK4(3) with adaptive stepping (best efficiency)  
🚀 **SSFM**: Split-Step Fourier Method (fastest for exploration)

## Performance
⚡ Zero-allocation hot paths with pre-allocated buffers  
⚡ Optimized FFT operations (FFTW with power-of-2 grids)  
⚡ Vectorized operations (@. macro throughout)  
⚡ Type-stable, no conditionals in solver loops  
⚡ Compile-time dispatch for specialized solvers

# Quick Start

## Basic Soliton Simulation
```julia
using JuGNLSE

# Step 1: Create time-frequency grid
grid = create_grid(2^12, 10e-12, 835e-9)  # 4096 pts, 10 ps, 835 nm

# Step 2: Define fiber medium
medium = Medium(
    0.15,        # 15 cm fiber
    0.11,        # γ = 0.11 W⁻¹m⁻¹
    [-11.83e-27, 8.03e-41],  # β₂, β₃
    0.0,         # lossless
    835e-9       # λ₀
)

# Step 3: Calculate N=1 soliton
beta2 = medium.betas[1]
T0 = 28.4e-15  # 1/e width
P0 = abs(beta2) / (medium.gamma * T0^2)

# Step 4: Create pulse
pulse = sech_pulse(grid, T0, P0, 835e-9, T0=true)

# Step 5: Configure simulation
params = SimParams(medium=medium, raman=false, shock=false)

# Step 6: Solve!
results = solve(pulse, params, method=:rk4ip)
```

## Supercontinuum Generation
```julia
# High-power N=10 soliton
pulse = sech_pulse(grid, T0, 10*P0, 835e-9, T0=true)

params = SimParams(
    medium = medium,
    N = 2^14,            # High resolution
    n_saves = 500,
    raman = true,        # Essential!
    shock = true,
    raman_model = Hollenbeck()
)

results = solve(pulse, params, method=:erk4ip, rtol=1e-7)
```

# Main Exports

## Core Types
- `Medium`: Fiber parameters (length, γ, [β₂, β₃, ...], α, λ₀)
- `Grid`: Time-frequency grid (t, ω, dt, dω)
- `Pulse`: Optical pulse (At, Aw)
- `SimParams`: Simulation configuration
- `RamanModel`: `BlowWood`, `LinAgrawal`, `Hollenbeck`

## Pulse Creation
- `sech_pulse`: Hyperbolic secant (solitons)
- `gaussian_pulse`: Gaussian profile
- `cw_pulse`: Continuous wave
- `custom_pulse`: Arbitrary shape

## Grid Creation
- `create_grid`: Manual grid specification
- `create_grid_from_medium`: Auto-estimate from medium

## Solvers
- `solve`: Main interface (method=:rk4ip, :erk4ip, or :ssfm)
- `propagate_rk4ip`: Direct RK4IP call
- `propagate_erk4ip`: Direct ERK4IP call
- `propagate_ssfm`: Direct SSFM call

## Utilities
- `calculate_soliton_power`: P₀ = |β₂|/(γT₀²)
- `soliton_order`: N = √(γP₀T₀²/|β₂|)
- `pulse_energy`: E = ∫|A(t)|² dt
- `peak_power`: max|A(t)|²
- `spectral_bandwidth`: Δω at FWHM
- `time_bandwidth_product`: TBP = Δt·Δν
- `fwhm`: Generic FWHM calculator

## Operators (Advanced Users)
- `dispersion_operator`: Linear operator D̂(ω)
- `apply_dispersion!`: In-place dispersion step
- `raman_response`: Calculate h_R(t)
- `raman_response_frequency`: Transform to R̃(ω)
- `nonlinear_operator`: N̂[A] for scalar γ
- `nonlinear_operator_frequency_dependent`: N̂[A] for γ(ω)

# Documentation

Access full documentation with examples:
```julia
?solve              # Main solver interface
?sech_pulse         # Pulse creation
?SimParams          # Parameter configuration
?Medium             # Fiber definition
```

# Performance Tips

1. **Use power-of-2 grids**: N = 2^12 or 2^14 for FFT efficiency
2. **Choose right method**: SSFM (fast preview) → ERK4IP (production) → RK4IP (benchmark)
3. **Optimize tolerances**: `reltol=1e-6` sufficient for most, `1e-8` for high accuracy
4. **Save strategically**: More `n_saves` → more memory, finer z-resolution
5. **Pre-allocate**: Avoid calling `solve()` in tight loops

# References

**GNLSE Theory:**  
G. P. Agrawal, "Nonlinear Fiber Optics," 6th ed. (Academic Press, 2019)

**Numerical Methods:**  
J. Hult, J. Lightwave Technol. 25, 3770 (2007) - RK4IP method

**M-GNLSE:**  
J. Lægsgaard, Opt. Express 15, 16110 (2007) - Pseudo-envelope method

**Raman Models:**  
K. J. Blow & D. Wood, IEEE J. Quantum Electron. 25, 2665 (1989)  
Q. Lin & G. P. Agrawal, Opt. Lett. 31, 3086 (2006)  
D. Hollenbeck & C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002)

# Contributing

Contributions welcome! Maintain:
- Zero allocations in hot paths
- Comprehensive docstrings with examples
- Unit tests for new features
- Type stability throughout
"""
module JuGNLSE

using FFTW
using LinearAlgebra
using OrdinaryDiffEq

# Include submodules
include("types.jl")
include("grid.jl")
include("pulses.jl")
include("dispersion.jl")
include("raman.jl")
include("nonlinearity.jl")

# Include solvers
include("solvers/ssfm.jl")
include("solvers/rk4ip.jl")
include("solvers/erk4ip.jl")
include("solver.jl")

include("utils.jl")

# Export types
export Medium, SimParams, Grid, Pulse
export RamanModel, BlowWood, LinAgrawal, Hollenbeck

# Export grid functions
export create_grid, create_grid_from_medium

# Export pulse functions
export sech_pulse, gaussian_pulse, cw_pulse, custom_pulse

# Export dispersion functions
export dispersion_operator, apply_dispersion, apply_dispersion!

# Export Raman functions
export raman_response, raman_response_frequency

# Export nonlinearity functions
export nonlinear_operator, nonlinear_operator_frequency_dependent, apply_nonlinearity!

# Export solver functions
export solve, propagate_ssfm, propagate_rk4ip, propagate_erk4ip

# Export utility functions
export calculate_soliton_power, soliton_order, pulse_energy, peak_power
export spectral_bandwidth, time_bandwidth_product, fwhm
export db_to_linear, linear_to_db
export wavelength_to_frequency, frequency_to_wavelength

end
