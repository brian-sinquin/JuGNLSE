module JuGNLSE

using FFTW
using LinearAlgebra
using Random

# Load core modules in order
include("core.jl")
include("propagation_steps.jl")

# Solvers
include("erk4ip.jl")
include("rk4.jl")
include("solver.jl") 
include("analysis.jl")

# Exports
export Medium, SimParams, Grid, Pulse, Solution
export RamanModel, BlowWood, LinAgrawal, Hollenbeck
export DispersionModel, TaylorDispersion, TabulatedDispersion
export AbstractGammaCoefficient, ConstantGamma, ZDependentGamma, WavelengthDependentGamma, GNLSEProblem
export AbstractGNLSESolver
export AbstractPropagationStep, Fiber, Loss, Filter, Amplifier, propagate!
export PhysicsModel
export create_grid, wavelength_grid
export sech_pulse, gaussian_pulse, lorentzian_pulse, cw_pulse
export dispersion_operator, propagation_constant
export raman_response
export gamma
export solve
export ERK4IP, RK4
export build_physics_model
export pulse_energy, peak_power, fwhm, spectral_bandwidth, time_bandwidth_product
export photon_number, spectral_centroid
export dispersion_length, nonlinear_length, soliton_number
export add_noise, rin_rms, spectral_coherence
export c

end # module
