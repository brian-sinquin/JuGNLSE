"""
    JuGNLSE

Numerical solver for the Generalized Nonlinear Schrödinger Equation (GNLSE) describing
optical pulse propagation in nonlinear dispersive media.

# Physical Effects

  - Dispersion: Arbitrary-order Taylor expansion β₂, β₃, β₄, ...
  - Kerr nonlinearity: Self-phase modulation, γ|A|²
  - Raman scattering: Delayed nonlinear response (BlowWood, LinAgrawal, Hollenbeck models)
  - Self-steepening: Shock term ∂(|A|²)/∂t for sub-100 fs pulses
  - Fiber loss: Constant α or frequency-dependent α(ω)
  - M-GNLSE: Frequency-dependent γ(ω) via pseudo-envelope method

# Solvers

  - `solve()`: Adaptive ERK4IP (embedded RK4(5) in interaction picture)
  - `propagate_ssfm()`: Symmetric split-step Fourier method (2nd order)
  - `propagate_rk4ip()`: Fixed-step RK4 in interaction picture

# Usage

```julia
using JuGNLSE

# Define grid and medium
grid = create_grid(2^12, 10e-12, 835e-9)
medium = Medium(0.15, 0.11, [-11.83e-27], 0.0, 835e-9)

# Create pulse and solve
pulse = sech_pulse(grid, 28.4e-15, 1e3)
params = SimParams(medium=medium)
results = solve(pulse, params)
```

# Main Exports

**Types**: `Medium`, `Grid`, `Pulse`, `SimParams`, `RamanModel`, `BlowWood`, `LinAgrawal`, `Hollenbeck`

**Pulses**: `sech_pulse`, `gaussian_pulse`, `cw_pulse`, `custom_pulse`

**Grids**: `create_grid`, `create_grid_from_medium`

**Solvers**: `solve`, `propagate_erk4ip`, `propagate_rk4ip`, `propagate_ssfm`

**Analysis**: `pulse_energy`, `peak_power`, `spectral_bandwidth`, `time_bandwidth_product`, `fwhm`

**Physics**: `calculate_soliton_power`, `soliton_order`, `dispersion_length`, `nonlinear_length`

**Operators**: `dispersion_operator`, `apply_dispersion!`, `raman_response`, `nonlinear_operator`

# References

G. P. Agrawal, "Nonlinear Fiber Optics" (Academic Press, 2019)
J. Hult, J. Lightwave Technol. 25, 3770 (2007)
J. Lægsgaard, Opt. Express 15, 16110 (2007)
"""
module JuGNLSE

using FFTW
using LinearAlgebra
using Unitful
using PhysicalConstants.CODATA2018: SpeedOfLightInVacuum

# Physical constants (extract numerical value in m/s)
const SPEED_OF_LIGHT = ustrip(u"m/s", SpeedOfLightInVacuum)  # 299792458.0 m/s

# Include submodules
include("types.jl")
include("grid.jl")
include("pulses.jl")
include("dispersion.jl")
include("raman.jl")
include("nonlinearity.jl")

# Solvers
include("solvers/erk4ip.jl")
include("solvers/rk4ip.jl")
include("solvers/ssfm.jl")

include("solver.jl")

include("physics.jl")
include("analysis.jl")

# Export types
export Medium, SimParams, Grid, Pulse
export RamanModel, BlowWood, LinAgrawal, Hollenbeck
export PhysicsModel  # Internal physics model struct

# Export grid functions
export create_grid, create_grid_from_medium

# Export pulse functions
export sech_pulse, gaussian_pulse, cw_pulse, custom_pulse

# Export dispersion functions
export dispersion_operator, apply_dispersion, apply_dispersion!

# Export Raman functions
export raman_response, raman_response_frequency


# Export solver functions
export solve
export propagate_erk4ip, propagate_rk4ip, propagate_ssfm

# Export utility functions
export calculate_soliton_power, soliton_order, pulse_energy, peak_power
export dispersion_length, nonlinear_length, soliton_period
export spectral_bandwidth, time_bandwidth_product, fwhm
export db_to_linear, linear_to_db
export wavelength_to_frequency, frequency_to_wavelength
export gamma_from_aeff, gamma_from_aeff_vec, aeff_from_measured_data
export convert_loss

end
