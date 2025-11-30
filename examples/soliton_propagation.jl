# Example: Fundamental Soliton Propagation
# Demonstrates propagation of N=1 soliton in optical fiber
# Solitons are pulses where dispersion and nonlinearity balance perfectly

using JuGNLSE
using Plots
using FFTW: fftshift

# Physical parameters
λ0 = 1550e-9      # Center wavelength [m]
β2 = -20e-27      # Group velocity dispersion [s²/m] (anomalous)
γ = 0.01          # Nonlinear coefficient [1/(W·m)]
T0 = 50e-15       # Pulse duration (1/e half-width) [s]

# Calculate fundamental soliton power
# For N=1 soliton: P0 = |β₂|/(γ*T0²)
P0 = abs(β2) / (γ * T0^2)
println("N=1 Soliton power: P0 = $(P0) W")

# Dispersion length: LD = T0²/|β₂|
LD = T0^2 / abs(β2)
println("Dispersion length: LD = $(LD*1000) mm")

# Propagation distance (2 dispersion lengths)
z_prop = 2.0 * LD
println("Propagation: $(z_prop*1000) mm = 2*LD\n")

# Create time-frequency grid
# Note: Wide enough to capture pulse evolution
grid = create_grid(2^12, 20e-12, λ0)

# Define medium
medium = Medium(z_prop, γ, [β2], 0.0, λ0)

# Simulation parameters
params = SimParams(
    medium=medium,
    N=2^12,
    n_saves=50,
    raman=false,
    shock=false
)

# Create soliton pulse
# IMPORTANT: Use T0=true to specify 1/e half-width (not FWHM)
# For soliton calculations, always use T0 since P0 depends on T0: P0 = |β₂|/(γ*T0²)
pulse = sech_pulse(grid, T0, P0, λ0, T0=true)

# Propagate
println("Propagating N=1 soliton with SSFM...")
results = solve(pulse, params, method=:ssfm, progress=true)

# Analyze results
E_initial = pulse_energy(results.At[:,1], grid.dt)
E_final = pulse_energy(results.At[:,end], grid.dt)
P_initial = maximum(abs2.(results.At[:,1]))
P_final = maximum(abs2.(results.At[:,end]))

println("\nResults:")
println("  Initial energy: $(E_initial*1e12) pJ")
println("  Final energy: $(E_final*1e12) pJ")
println("  Energy change: $((E_final - E_initial)/E_initial * 100) %")
println("  Initial peak power: $(P_initial) W")
println("  Final peak power: $(P_final) W")
println("  Peak power change: $((P_final - P_initial)/P_initial * 100) %")

# Shape fidelity
initial_shape = abs2.(results.At[:,1])
final_shape = abs2.(results.At[:,end])
initial_norm = initial_shape / maximum(initial_shape)
final_norm = final_shape / maximum(final_shape)
fidelity = sum(sqrt.(initial_norm .* final_norm)) / sqrt(sum(initial_norm) * sum(final_norm))
println("  Shape fidelity: $(fidelity * 100) %")

# Visualization
t_ps = grid.t * 1e12
z_mm = results.z * 1000
f_THz = fftshift(grid.omega) / (2π*1e12)
E_z = [pulse_energy(results.At[:,i], grid.dt) for i in 1:size(results.At, 2)]

# Create 2x2 layout
p1 = heatmap(t_ps, z_mm, abs2.(results.At)', 
    xlabel="Time (ps)", ylabel="Distance (mm)",
    title="Soliton Temporal Evolution", c=:viridis)

p2 = plot(t_ps, abs2.(results.At[:,1]), label="Initial", lw=2,
    xlabel="Time (ps)", ylabel="Power (W)",
    title="Initial vs Final Pulse")
plot!(p2, t_ps, abs2.(results.At[:,end]), label="Final", lw=2, ls=:dash)

p3 = heatmap(f_THz, z_mm, fftshift(abs2.(results.Aw), 1)', 
    xlabel="Frequency offset (THz)", ylabel="Distance (mm)",
    title="Spectral Evolution", c=:viridis, xlims=(-15, 15))

p4 = plot(z_mm, (E_z .- E_z[1]) ./ E_z[1] .* 100, lw=2,
    xlabel="Distance (mm)", ylabel="Relative energy change (%)",
    title="Energy Conservation", legend=false)
hline!(p4, [0], color=:red, ls=:dash)

fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
display(fig)
savefig("soliton_propagation.png")
println("\nFigure saved as soliton_propagation.png")
