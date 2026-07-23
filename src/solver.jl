"""
Main solver interface following gnlse-python conventions.

Reference: gnlse-python GNLSE.run()
"""

# Access the Solvers submodule defined in solvers/interface.jl
import .Solvers
using .Solvers: AbstractGNLSESolver, ERK4IP # Import specific types/constructors

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
    # Delegate the call to the solve function defined within the Solvers module
    return Solvers.solve(problem, solver; progress=progress)
end
