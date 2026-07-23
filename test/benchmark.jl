using Pkg
Pkg.activate(expanduser("~/JuGNLSE"))
using BenchmarkTools
using JuGNLSE

# --- Gold Standard PCF Supercontinuum Generation Parameters ---
const N_POINTS = 2^10 # Réduit pour benchmark rapide
const TIME_WINDOW = 12.5e-12 
const Z_SAVES = 200

const WAVELENGTH = 835e-9 
const FIBER_LENGTH = 0.15 
const NONLINEARITY = 0.11 
const LOSS = 0.0 
const BETAS = [-11.83e-27] # SI units: s^2/m

const PEAK_POWER = 10000.0 
const DURATION = 50e-15 

# --- JuGNLSE Setup ---
function run_full_sim()
    grid = create_grid(N_POINTS, TIME_WINDOW, WAVELENGTH)
    medium = Medium(FIBER_LENGTH, NONLINEARITY, LOSS, BETAS, WAVELENGTH)
    pulse = sech_pulse(grid, PEAK_POWER, DURATION)
    params = SimParams(medium=medium, z_saves=Z_SAVES, raman_model=BlowWood())
    problem = GNLSEProblem(medium=medium, grid=grid, initial_pulse=pulse, sim_params=params)
    return solve(problem; progress=false)
end

# --- Benchmark ---
println("Starting JuGNLSE Benchmark...")
# Warm-up
run_full_sim()

# Benchmark
bench = @benchmark run_full_sim() samples=5 evals=1
display(bench)
