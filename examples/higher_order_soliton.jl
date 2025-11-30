# Example: Higher-Order Soliton (N=2)
# Demonstrates periodic compression and expansion of N=2 soliton
# Higher-order solitons exhibit periodic breathing behavior

using JuGNLSE
using Plots
using FFTW: fftshift
# Physical parameters
λ0 = 1550e-9      # Center wavelength [m]
β2 = -20e-27      # Group velocity dispersion [s²/m] (anomalous)
γ = 0.01          # Nonlinear coefficient [1/(W·m)]
T0 = 50e-15       # Pulse duration (1/e half-width) [s]

# Calculate N=2 soliton power
# For N-th order soliton: P0 = N² * |β₂|/(γ*T0²)
N_soliton = 2
P0 = N_soliton^2 * abs(β2) / (γ * T0^2)
println("N=$(N_soliton) Soliton power: P0 = $(P0) W")

# Dispersion length and soliton period
LD = T0^2 / abs(β2)
z_period = π / 2 * LD  # Period for higher-order solitons
println("Dispersion length: LD = $(LD*1000) mm")
println("Soliton period: z0 = $(z_period*1000) mm = π/2 * LD")

# Propagate 1.5 periods to see periodic behavior
z_prop = 1.5 * z_period
println("Propagation: $(z_prop*1000) mm = 1.5 periods\n")

# Create time-frequency grid
grid = create_grid(2^12, 20e-12, λ0)

# Define medium
medium = Medium(z_prop, γ, [β2], 0.0, λ0)

# Simulation parameters
params = SimParams(
    medium=medium,
    N=2^12,
    n_saves=100,  # More saves to capture periodic behavior
    raman=false,
    shock=false,
)

# Create N=2 soliton pulse
# IMPORTANT: Use T0=true for correct soliton power calculation
pulse = sech_pulse(grid, T0, P0, λ0, T0=true)

# Propagate
println("Propagating N=$(N_soliton) soliton with ERK4IP...")
results = solve(pulse, params, method=:erk4ip, progress=true)

# Analyze peak power evolution
P_peak = [maximum(abs2.(results.At[:,i])) for i in 1:size(results.At, 2)]
P_compression = maximum(P_peak) / P_peak[1]
println("\nResults:")
println("  Initial peak power: $(P_peak[1]) W")
println("  Maximum peak power: $(maximum(P_peak)) W")
println("  Compression ratio: $(P_compression)x")
println("  At z = $(results.z[argmax(P_peak)]*1000) mm")

# Find period from peak power oscillations
# (simple analysis - just check if returns to initial)
P_final = P_peak[end]
periodic_return = abs(P_final - P_peak[1]) / P_peak[1] * 100
println("  Periodic return error: $(periodic_return) %")

# Energy conservation
E_z = [pulse_energy(results.At[:,i], grid.dt) for i in 1:size(results.At, 2)]
E_drift = abs(E_z[end] - E_z[1]) / E_z[1] * 100
println("  Energy drift: $(E_drift) %")

# Visualization
t_ps = grid.t * 1e12
z_mm = results.z * 1000
f_THz = fftshift(grid.omega) / (2π*1e12)

# Calculate FWHM evolution
fwhm_z = Float64[]
for i in 1:size(results.At, 2)
    P = abs2.(results.At[:,i])
    P_norm = P / maximum(P)
    idx_above = findall(P_norm .> 0.5)
    if !isempty(idx_above)
        fwhm = (idx_above[end] - idx_above[1]) * grid.dt * 1e15
        push!(fwhm_z, fwhm)
    else
        push!(fwhm_z, NaN)
    end
end

# Find key indices
idx_max = argmax(P_peak)
idx_period = argmin(abs.(results.z .- z_period))

# Create plots
p1 = heatmap(t_ps, z_mm, abs2.(results.At)', 
    xlabel="Time (ps)", ylabel="Distance (mm)",
    title="N=$(N_soliton) Soliton - Periodic Compression", c=:viridis,
    xlims=(-0.5, 0.5))

p2 = plot(z_mm, P_peak, lw=2, label="Peak power",
    xlabel="Distance (mm)", ylabel="Peak Power (W)",
    title="Periodic Breathing")
hline!(p2, [P_peak[1]], color=:red, ls=:dash, label="Initial")
vline!(p2, [z_period*1000], color=:green, ls=:dash, label="Period")

p3 = plot(z_mm, fwhm_z, lw=2, color=:orange, legend=false,
    xlabel="Distance (mm)", ylabel="FWHM (fs)",
    title="Pulse Width Oscillation")
hline!(p3, [fwhm_z[1]], color=:red, ls=:dash)

p4 = heatmap(f_THz, z_mm, fftshift(abs2.(results.Aw), 1)', 
    xlabel="Frequency offset (THz)", ylabel="Distance (mm)",
    title="Spectral Evolution", c=:plasma, xlims=(-30, 30))

p5 = plot(t_ps, abs2.(results.At[:,1]), label="z = 0", lw=2,
    xlabel="Time (ps)", ylabel="Power (W)",
    title="Snapshots at Key Points")
plot!(p5, t_ps, abs2.(results.At[:,idx_max]), label="Max compression", lw=2)
plot!(p5, t_ps, abs2.(results.At[:,idx_period]), label="One period", lw=2, ls=:dash)

fig = plot(p1, p2, p3, p4, p5, layout=@layout([a{0.5w} [b; c]; d{0.5w} e]), 
    size=(1400, 1000))
display(fig)
savefig("higher_order_soliton.png")
println("\nFigure saved as higher_order_soliton.png")
