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

```julia
using JuGNLSE

# Define the physical medium
medium = Medium(0.15, 0.11, [-11.83e-27, 8.13e-41], 0.0, 835e-9)

# Set up the computational grid
grid = create_grid(2^12, 10e-12, 835e-9)

# Generate an initial pulse
pulse = sech_pulse(grid, 50e-15, 10000.0)

# Configure and solve
params = SimParams(medium=medium, raman=true, shock=true)
results = solve(pulse, params)
```

## Documentation

For detailed information on the physical models, numerical methods, and API reference, please refer to the [documentation](https://brian-sinquin.github.io/JuGNLSE.jl/).
