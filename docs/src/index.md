```@meta
CurrentModule = JuGNLSE
```

# JuGNLSE.jl

Documentation for [JuGNLSE.jl](https://github.com/brian-sinquin/JuGNLSE.jl), a high-performance Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE).

## Overview

JuGNLSE.jl provides efficient tools for modeling ultrafast nonlinear pulse propagation in optical fibers. The package implements state-of-the-art numerical methods including ERK4(3)-IP (Embedded Runge-Kutta 4(3) in Interaction Picture), RK4IP (Runge-Kutta 4th order in Interaction Picture), and Split-Step Fourier Method (SSFM) with comprehensive physics models.

## Features

- **Advanced Integration Methods**
  - ERK4(3)-IP: Embedded Runge-Kutta with adaptive stepping and error control
  - RK4IP: Runge-Kutta 4th order in Interaction Picture
  - SSFM: Split-Step Fourier Method for fast fixed-step propagation
  - Automatic error control and step size adaptation

- **Complete Physics Models**
  - Higher-order dispersion (arbitrary order)
  - Kerr nonlinearity (SPM, XPM)
  - Self-steepening (shock term)
  - Stimulated Raman scattering (3 models)
  - Frequency-dependent loss

- **Flexible and Fast**
  - Type-stable implementations
  - Pre-planned FFTs
  - ~5-10x faster than Python
  - Optimized for Julia 1.6+

## Installation

Install JuGNLSE from the Julia package manager:

```julia
using Pkg
Pkg.add("JuGNLSE")
```

For development:

```julia
Pkg.develop(url="https://github.com/brian-sinquin/JuGNLSE.jl")
```

## Quick Start Example

```julia
using JuGNLSE

# Define fiber (medium) parameters
medium = Medium(
    0.15,                              # Length: 15 cm
    0.11,                              # Nonlinear coefficient γ [W⁻¹m⁻¹]
    [0.0, 0.0, -11.83e-27, 8.13e-41], # Dispersion βₙ [psⁿ/m]
    0.0,                               # Loss α [dB/km]
    835e-9                             # Center wavelength [m]
)

# Create time-frequency grid
grid = create_grid(2^12, 10e-12, 835e-9)

# Generate input pulse (50 fs sech, 10 kW)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)

# Setup simulation
params = SimParams(
    medium=medium,
    N=2^12,
    n_saves=200,
    raman=true,
    shock=true,
    raman_model=Hollenbeck()
)

# Solve GNLSE with ERK4(3)-IP (adaptive stepping)
results = solve(pulse, params, method=:erk4ip)

# Results contain:
# results.z   - Propagation distances
# results.At  - Time-domain evolution
# results.Aw  - Frequency-domain evolution
```

## Package Structure

### Core Types

- `Medium`: Fiber/medium optical properties (β coefficients, γ, α, λ₀)
- `SimParams`: Simulation configuration (grid size, saves, physics options)
- `Grid`: Time-frequency grid with FFT planning
- `Pulse`: Pulse envelope representation (time and frequency domain)

### Raman Models

- `BlowWood`: Blow-Wood (1989) single Lorentzian
- `LinAgrawal`: Lin-Agrawal (2006) with Boson peak
- `Hollenbeck`: Hollenbeck-Cantrell (2002) 13-oscillator

### Main Functions

- `create_grid`: Initialize computational grid
- `sech_pulse`, `gaussian_pulse`, `cw_pulse`: Pulse generation
- `solve`: Main GNLSE solver interface
- `dispersion_operator`: Linear dispersion
- `raman_response`: Raman response functions

## Physical Background

### GNLSE Equation

The generalized nonlinear Schrödinger equation describes pulse propagation in optical fibers including dispersion, nonlinearity, and loss effects.

### Performance Tips

1. **Use power-of-2 grid sizes** for optimal FFT performance
2. **Pre-allocate** arrays for multiple simulations
3. **Choose appropriate tolerances**: Default `rtol=1e-6`, `atol=1e-8` for ERK4IP
4. **Solver selection**:
   - **ERK4IP**: Best for adaptive stepping with automatic error control (recommended)
   - **RK4IP**: Fixed-step with OrdinaryDiffEq.jl integration (more control)
   - **SSFM**: Fastest for fixed step size, lower accuracy for highly nonlinear problems
5. **ERK4IP typically uses fewer steps** than RK4IP while maintaining accuracy

## References

1. Agrawal, G. P. (2012). *Nonlinear Fiber Optics* (5th ed.). Academic Press.
2. Dudley, J. M., Genty, G., & Coen, S. (2006). Supercontinuum generation in photonic crystal fiber. *Reviews of Modern Physics*, 78(4), 1135.
3. Hult, J. (2007). A fourth-order Runge–Kutta in the interaction picture method. *Journal of Lightwave Technology*, 25(12), 3770-3775.

## Index

```@index
```

## API Reference

```@autodocs
Modules = [JuGNLSE]
```
