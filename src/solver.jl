"""
Main solver interface following gnlse-python conventions.

Reference: gnlse-python GNLSE.run()
"""

using .Solvers: AbstractGNLSESolver, ERK4IP

"""
    solve(problem::GNLSEProblem, solver::AbstractGNLSESolver=ERK4IP(); progress::Bool=true)

Solves the Generalized Nonlinear Schrödinger Equation (GNLSE) for a given `GNLSEProblem`.
This is the main entry point for running a simulation.

# Arguments
  - `problem::GNLSEProblem`: A `GNLSEProblem` object containing all necessary simulation parameters (medium, grid, initial pulse, simulation parameters, and gamma coefficient).
  - `solver::AbstractGNLSESolver=ERK4IP()`: The solver to use for propagation. Defaults to `ERK4IP`.
  - `progress::Bool=true`: If `true`, a progress bar will be displayed during the simulation.

# Returns
  - `Solution`: A `Solution` object containing the pulse's evolution through the fiber.

# Example

```julia
using JuGNLSE

# Define grid (natural SI units: s, m, W)
grid = create_grid(2^13, 12.5e-12, 835e-9)  # resolution, time_window [s], λ [m]

# Define medium: Medium(L[m], γ[1/W/m], loss[dB/m], betas[sⁿ/m], λ[m])
medium = Medium(0.15, 0.11, 0.0, [-11.83e-27], 835e-9)

# Create pulse
pulse = sech_pulse(grid, 10000.0, 50e-15)  # Pmax [W], FWHM [s]

# Setup simulation parameters
sim_params = SimParams(;
    medium=medium,
    z_saves=200,
    raman_model=BlowWood(),
    self_steepening=false,
    rtol=1e-6,
    atol=1e-8,
)

# Create the GNLSE problem
problem = GNLSEProblem(medium=medium, grid=grid, initial_pulse=pulse, sim_params=sim_params)

# Solve the problem
solution = solve(problem)
```

# Notes
Integrates the GNLSE with the specified solver. All quantities are in natural SI units; the envelope spectrum follows the standard optics convention `AW = ifft(At)`.

Photon number conservation is checked for lossless fibers, and a warning is issued if significant drift occurs, suggesting a tighter `rtol`/`atol`.
"""
function solve(problem::GNLSEProblem, solver::AbstractGNLSESolver=ERK4IP(); progress::Bool=true)
    return Solvers.solve(problem, solver; progress=progress)
end
