"""
Main solver interface following gnlse-python conventions.
"""

"""
    solve(problem::GNLSEProblem, solver::AbstractGNLSESolver=ERK4IP(); progress::Bool=true)

Generic fallback `solve` function.
"""
function solve(problem::GNLSEProblem, solver::AbstractGNLSESolver; progress::Bool=true)
    error("solve not implemented for $(typeof(solver))")
end
