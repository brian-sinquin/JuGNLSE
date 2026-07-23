module Solvers

using FFTW
using LinearAlgebra
using ProgressMeter
using ..JuGNLSE: GNLSEProblem, Solution, Pulse, SimParams, photon_number

# Re-import types needed for solver interface
import ..JuGNLSE: AbstractGNLSESolver

export AbstractGNLSESolver, ERK4IP, RK4

function solve(problem::GNLSEProblem, solver::AbstractGNLSESolver; progress::Bool=true)
    error("solve not implemented for $(typeof(solver))")
end

include("erk4ip.jl")
include("rk4.jl")

end # module Solvers
