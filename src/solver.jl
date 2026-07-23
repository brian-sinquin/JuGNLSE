"""
Main solver interface following gnlse-python conventions.

Reference: gnlse-python GNLSE.run()
"""

"""
    solve(problem::GNLSEProblem; progress::Bool=true)
    solve(problem::GNLSEProblem, solver::AbstractGNLSESolver; progress::Bool=true)

Solves the Generalized Nonlinear Schrödinger Equation (GNLSE) for a given `GNLSEProblem`.
This is the main entry point for running a simulation.

# Arguments
  - `problem::GNLSEProblem`: A `GNLSEProblem` object containing all necessary simulation parameters.
  - `solver::AbstractGNLSESolver=ERK4IP()`: The solver to use for propagation. Defaults to `ERK4IP`.
  - `progress::Bool=true`: If `true`, a progress bar will be displayed during the simulation.

# Returns
  - `Solution`: A `Solution` object containing the pulse's evolution through the fiber.
"""
function solve(problem::GNLSEProblem; progress::Bool=true)
    return solve(problem, ERK4IP(); progress=progress)
end
