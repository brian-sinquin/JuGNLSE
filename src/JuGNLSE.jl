"""
    JuGNLSE

A Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE)
for modeling nonlinear pulse propagation in optical waveguides and nonlinear media.

# Features
- RK4IP (Runge-Kutta 4th order in Interaction Picture) with adaptive stepping
- ERK4IP (Embedded RK4 in Interaction Picture) with adaptive step-size control
- Split-Step Fourier Method (SSFM)
- Multiple Raman response models (Blow-Wood, Lin-Agrawal, Hollenbeck)
- Self-steepening (shock) effects
- Arbitrary pulse shapes (Sech, Gaussian, CW, custom)
- Frequency-dependent dispersion and loss

# Main exports
- Types: `Medium`, `SimParams`, `Grid`, `Pulse`
- Raman models: `BlowWood`, `LinAgrawal`, `Hollenbeck`
- Functions: `create_grid`, `sech_pulse`, `gaussian_pulse`, `solve`

# Example
```julia
using JuGNLSE

# Create nonlinear medium
medium = Medium(
    0.15,                          # length [m]
    0.11,                          # gamma [W⁻¹m⁻¹]
    [0.0, 0.0, -11.83e-27, 8.13e-41],  # betas [s^n/m]
    0.0,                           # alpha [dB/km]
    835e-9                         # lambda0 [m]
)

# Create grid
grid = create_grid(2^12, 10e-12, 835e-9)

# Create pulse
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

# Solve
results = solve(pulse, params)
```
"""
module JuGNLSE

using FFTW
using LinearAlgebra
using OrdinaryDiffEq
using Reexport

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
