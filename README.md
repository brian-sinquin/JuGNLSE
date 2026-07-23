# JuGNLSE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brian-sinquin.github.io/JuGNLSE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brian-sinquin.github.io/JuGNLSE.jl/dev/)
[![Build status](https://github.com/brian-sinquin/JuGNLSE/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/brian-sinquin/JuGNLSE/actions/workflows/CI.yml)

JuGNLSE.jl is a Julia package for solving the Generalized Nonlinear Schrödinger Equation (GNLSE). It models optical pulse propagation in nonlinear media (e.g., optical fibers) with high numerical stability and performance.

## New Features
- **Flexible Gamma**: Support for wavelength and propagation-dependent (`z`) nonlinear coefficients.
- **Pipeline API**: Composable simulation steps (Fiber, Loss, Amplifier) via `propagate!`.
- **Extensible Solvers**: Interface-based design for adding arbitrary numerical solvers.

## Installation

```julia
using Pkg
Pkg.add("JuGNLSE")
```

## Basic Usage

### Standard Solver
```julia
using JuGNLSE

medium = Medium(0.15, 0.11, 0.0, [-11.83e-27], 835e-9)
grid = create_grid(2^10, 10e-12, 835e-9)
pulse = sech_pulse(grid, 10000.0, 50e-15)
problem = GNLSEProblem(medium=medium, grid=grid, initial_pulse=pulse)

solution = solve(problem)
```

### Composable Pipeline
```julia
using JuGNLSE

pipeline = [
    Fiber(medium, 0.1),
    Loss(0.5),
    Amplifier(3.0),
    Fiber(medium, 0.05)
]

solution = propagate!(pulse, pipeline)
```

## Documentation

For detailed information on the physical models, numerical methods, and API reference, please refer to the [documentation](https://brian-sinquin.github.io/JuGNLSE.jl/).
