# Simulate supercontinuum generation for parameters similar
# to Fig.3 of Dudley et. al, RMP 78 1135 (2006)
# Original MATLAB code by J.C. Travers, M.H Frosz and J.M. Dudley (2009)
# Adapted for JuGNLSE.jl
# Please cite Dudley et. al, RMP 78 1135 (2006) in any publication using this code.

using JuGNLSE
using Plots
using FFTW: fftshift

# Physical constants
c = 299792458.0             # Speed of light [m/s]

# === Numerical grid
N = 2^13                    # Number of grid points
twidth = 12.5e-12           # Width of time window [s]
λ0 = 835e-9                 # Reference wavelength [m]

# Create time-frequency grid
grid = create_grid(N, twidth, λ0)

# === Input pulse
P0 = 10000.0                # Peak power [W]
T0 = 28.4e-15               # Pulse duration (1/e half-width) [s]

# Create sech pulse (T0=true means T0 is 1/e half-width, not FWHM)
pulse = sech_pulse(grid, T0, P0, λ0, T0=true)

# === Fiber parameters
flength = 0.15              # Fiber length [m]
γ = 0.11                    # Nonlinear coefficient [1/W/m]
α = 0.0                     # Loss [dB/m]

# Dispersion coefficients [β₂, β₃, β₄, β₅, β₆, β₇, β₈, β₉, β₁₀]
# Units: [s^n/m] for βₙ
betas = [-1.1830e-26, 8.1038e-41, -9.5205e-56, 2.0737e-70,
         -5.3943e-85, 1.3486e-99, -2.5495e-114, 3.0524e-129,
         -1.7140e-144]

# Define medium
medium = Medium(flength, γ, betas, α, λ0)

# === Raman parameters
# The original MATLAB code uses Blow-Wood Raman model with:
# τ1 = 12.2 fs, τ2 = 32 fs, fr = 0.18
# These are exactly the default parameters in JuGNLSE's BlowWood() model
fr = 0.18                   # Fractional Raman contribution

# === Simulation parameters
params = SimParams(
    medium = medium,
    N = N,
    n_saves = 200,          # Number of z-points to save
    raman = true,           # Enable Raman effect
    shock = false,          # Disable self-steepening (not in original)
    raman_model = BlowWood(),  # Blow-Wood model (τ1=12.2fs, τ2=32fs)
    fr = fr
)

println("Starting supercontinuum simulation...")
println("Grid: $(N) points, $(twidth*1e12) ps window")
println("Pulse: P0 = $(P0) W, T0 = $(T0*1e15) fs")
println("Fiber: L = $(flength*100) cm, γ = $(γ) W⁻¹m⁻¹")
println("Raman: fr = $(fr), Blow-Wood model (τ1=12.2 fs, τ2=32 fs)")
println()

# Propagate field
results = solve(pulse, params, method=:rk4ip, progress=true)

# === Extract results
z = results.z               # Propagation distances [m]
At = results.At             # Time-domain field [sqrt(W)]
Aw = results.Aw             # Frequency-domain field [sqrt(W·s)]

# Calculate energy conservation
E_initial = pulse_energy(At[:,1], grid.dt)
E_final = pulse_energy(At[:,end], grid.dt)
println("\nEnergy conservation:")
println("  Initial: $(E_initial*1e12) pJ")
println("  Final: $(E_final*1e12) pJ")
println("  Change: $((E_final - E_initial)/E_initial * 100) %")

# === Plotting
println("\nGenerating plots...")

# Convert to convenient units
t_ps = grid.t * 1e12        # Time in ps
z_m = z                     # Distance in m
λ = 2π * c ./ fftshift(grid.omega .+ (2pi*c/λ0)) # Wavelength grid [m]
λ_nm = λ * 1e9 |> reverse  # Wavelength in nm

# Select wavelength range for plotting (450-1350 nm)
λ_range = (λ_nm .>= 450) .& (λ_nm .<= 1350)

# Spectral intensity in dB (normalized to wavelength)
# Convert |A(ω)|² to spectral density per wavelength: |A(ω)|² * (2πc/λ²)
spectral_density = fftshift(abs2.(Aw), 1) .* (2π * c) ./ (λ.^2)
lIW = 10 * log10.(spectral_density)
mlIW = maximum(lIW[λ_range, :])  # Max for color scaling

# Temporal intensity in dB
lIT = 10 * log10.(abs2.(At))
mlIT = maximum(lIT)

# Create figure with two subplots
p1 = heatmap(λ_nm[λ_range], z_m, lIW[λ_range, :]',
    xlabel="Wavelength (nm)", ylabel="Distance (m)",
    title="Supercontinuum Spectral Evolution",
    c=:hot, clims=(mlIW-40, mlIW), colorbar_title="dB")

p2 = heatmap(t_ps, z_m, lIT',
    xlabel="Time (ps)", ylabel="Distance (m)",
    title="Temporal Evolution",
    c=:hot, clims=(mlIT-40, mlIT), xlims=(-0.5, 5), colorbar_title="dB")

fig = plot(p1, p2, layout=(1,2), size=(1400, 500))
display(fig)
savefig("supercontinuum_generation.png")
println("Figure saved as supercontinuum_generation.png")

# Optional: Plot initial and final spectra
p3 = plot(λ_nm[λ_range], lIW[λ_range, 1], 
    label="Input", lw=2, xlabel="Wavelength (nm)", 
    ylabel="Spectral density (dB)", title="Spectra Comparison",
    xlims=(450, 1350))
plot!(p3, λ_nm[λ_range], lIW[λ_range, end], 
    label="Output (z=$(flength*100) cm)", lw=2)
display(p3)
savefig("supercontinuum_spectra.png")
println("Spectral comparison saved as supercontinuum_spectra.png")