"""
Main solver interface following gnlse-python conventions.

Reference: gnlse-python GNLSE.run()
"""

"""
    solve(problem::GNLSEProblem, solver::AbstractGNLSESolver=ERK4IP(); progress::Bool=true)

Solves the Generalized Nonlinear Schrödinger Equation (GNLSE) for a given `GNLSEProblem`.
This is the main entry point for running a simulation.

# Arguments
  - `problem::GNLSEProblem`: A `GNLSEProblem` object containing all necessary simulation parameters.
  - `solver::AbstractGNLSESolver=ERK4IP()`: The solver to use for propagation. Defaults to `ERK4IP`.
  - `progress::Bool=true`: If `true`, a progress bar will be displayed during the simulation.

# Returns
  - `Solution`: A `Solution` object containing the pulse's evolution through the fiber.
"""
function solve(problem::GNLSEProblem, solver::AbstractGNLSESolver=ERK4IP(); progress::Bool=true)
    # The Solvers module wrapper was removed. The `solve` function is now directly in JuGNLSE namespace.
    # We call the solve function defined in solvers/interface.jl
    return solve(problem, solver; progress=progress)
end
