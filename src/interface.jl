# Solver interface and implementations
# This file defines the solver interface and solvers, now directly in the JuGNLSE module.

# --- Interface ---
abstract type AbstractGNLSESolver end

function solve(problem::GNLSEProblem, solver::AbstractGNLSESolver; progress::Bool=true)
    error("solve not implemented for $(typeof(solver))")
end

# --- Implementations ---
include("erk4ip.jl")
include("rk4.jl")
