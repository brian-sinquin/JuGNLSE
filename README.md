# JuGNLSE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brian-sinquin.github.io/JuGNLSE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brian-sinquin.github.io/JuGNLSE.jl/dev/)
[![Build status](https://github.com/brian-sinquin/JuGNLSE/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/brian-sinquin/JuGNLSE/actions/workflows/CI.yml)

JuGNLSE.jl is a Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE). It is designed to model the propagation of optical pulses in nonlinear media, such as optical fibers, with a focus on performance and numerical stability.

## Installation

The package can be installed using the Julia package manager:

```julia
using Pkg
Pkg.add("JuGNLSE")
```

## Basic Usage

All quantities are in natural SI units (seconds, metres, watts).

```julia
using JuGNLSE

# Define the fiber medium (keyword constructor — avoids argument-order mistakes)
medium = Medium(;
    fiber_length = 0.15,                  # m
    gamma        = 0.11,                  # 1/(W·m)
    loss         = 0.0,                   # dB/m
    betas        = [-11.83e-27, 8.13e-41],# Taylor dispersion: β₂ [s²/m], β₃ [s³/m], …
    lambda0      = 835e-9,                # m
)

# Set up the time–frequency grid: resolution, time window [s], center wavelength [m]
grid = create_grid(2^13, 12.5e-12, 835e-9)

# Generate an initial pulse: peak power [W], FWHM [s]
pulse = sech_pulse(grid, 10000.0, 50e-15)

# Configure the simulation and solve
params = SimParams(; medium=medium, raman_model=BlowWood(), self_steepening=true)
solution = solve(pulse, params)

# `solution.At` / `solution.AW` are (N × z_saves); each column is one distance.
```

Measured dispersion can be supplied instead of a Taylor expansion via
`TabulatedDispersion(detuning, beta)` (passed as `dispersion=` to `Medium`).

## Documentation

For detailed information on the physical models, numerical methods, and API reference, please refer to the [documentation](https://brian-sinquin.github.io/JuGNLSE.jl/).
