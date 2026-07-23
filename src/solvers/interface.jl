# Abstract base type for GNLSE solvers.
abstract type AbstractGNLSESolver end

"""
    solve(problem::GNLSEProblem, solver::AbstractGNLSESolver; progress::Bool=true)

Generic fallback `solve` function for `AbstractGNLSESolver`s.
This method is intended to be overloaded by concrete solver implementations (e.g., `ERK4IP`, `RK4`).
If called without a specific solver implementation, it will throw an error.

# Arguments
  - `problem::GNLSEProblem`: The GNLSE problem to solve.
  - `solver::AbstractGNLSESolver`: The specific solver instance to use (e.g., `ERK4IP()`).
  - `progress::Bool=true`: Whether to display a progress bar during the solution process.

# Throws
`ERROR`: If the `solve` method is not implemented for the given `solver` type.
"""
function solve(problem::GNLSEProblem, solver::AbstractGNLSESolver; progress::Bool=true)
    error("solve not implemented for $(typeof(solver))")
end

include("erk4ip.jl")
include("rk4.jl")
