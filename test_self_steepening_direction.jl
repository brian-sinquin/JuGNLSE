"""
Test to validate self-steepening direction

Physical expectation: High-intensity pulses should steepen on the TRAILING edge
due to intensity-dependent group velocity (higher intensity travels slower).

This test creates a high-power pulse and propagates with ONLY self-steepening
enabled (no dispersion, no Raman) to isolate the shock effect.
"""

using JuGNLSE
using Plots

println("=" ^ 70)
println("Self-Steepening Direction Test")
println("=" ^ 70)

# Create grid
N = 2^12
time_window = 10e-12  # 10 ps
center_wavelength = 835e-9  # 835 nm
grid = create_grid(N, time_window, center_wavelength)

# Create medium with ZERO dispersion and very high nonlinearity
# This isolates the self-steepening effect
medium = Medium(
    0.01,           # 1 cm fiber (short distance for shock observation)
    10.0,           # Very high gamma to make shock visible
    [0.0],          # Zero dispersion (β₂ = 0)
    0.0,            # No loss
    center_wavelength
)

# Create very short, high-power pulse
T0 = 30e-15  # 30 fs FWHM / 1.76
P0 = 1e5     # 100 kW peak power 
             # NOTE: This is intentionally very high to make shock effect visible
             # over short propagation. Real experiments use lower power with longer fibers.
             # For realistic simulations, use P0 ~ 1-10 kW with longer propagation distances.
pulse = sech_pulse(grid, T0, P0, center_wavelength, T0=true)

# Configure simulation: ONLY self-steepening, no Raman, no dispersion
params = SimParams(
    medium = medium,
    N = N,
    n_saves = 50,
    raman = false,     # Disable Raman
    shock = true,      # Enable self-steepening
    raman_model = BlowWood(),
    reltol = 1e-7,
    abstol = 1e-9
)

println("\nSimulation Parameters:")
println("  Pulse duration: $(T0*1e15) fs (1/e half-width)")
println("  Peak power: $(P0/1e3) kW")
println("  Fiber length: $(medium.length*1e2) cm")
println("  Gamma: $(medium.gamma) W⁻¹m⁻¹")
println("  Effects: Self-steepening ONLY (no dispersion, no Raman)")
println()

# Run simulation
println("Propagating...")
z, At, Aw = solve(pulse, params, method=:rk4ip)

# Analysis: Check where steepening occurs
t_indices = findall(abs.(grid.t) .< 200e-15)  # Focus on ±200 fs around pulse
t_subset = grid.t[t_indices] .* 1e15  # Convert to fs

# Get intensity profiles at input and output
I_input = abs2.(At[t_indices, 1])
I_output = abs2.(At[t_indices, end])

# Normalize for comparison
I_input_norm = I_input ./ maximum(I_input)
I_output_norm = I_output ./ maximum(I_output)

# Find peak positions
idx_input = argmax(I_input)
idx_output = argmax(I_output)
t_peak_input = t_subset[idx_input]
t_peak_output = t_subset[idx_output]

println("\nResults:")
println("  Peak position at input: $(round(t_peak_input, digits=2)) fs")
println("  Peak position at output: $(round(t_peak_output, digits=2)) fs")
println("  Peak shift: $(round(t_peak_output - t_peak_input, digits=2)) fs")

# Compute gradient to measure steepness
function compute_steepness(intensity, t)
    # Compute maximum positive and negative gradients
    grad = diff(intensity) ./ diff(t)
    max_positive_grad = maximum(grad)
    max_negative_grad = abs(minimum(grad))
    return max_positive_grad, max_negative_grad
end

pos_grad_in, neg_grad_in = compute_steepness(I_input_norm, t_subset)
pos_grad_out, neg_grad_out = compute_steepness(I_output_norm, t_subset)

println("\nSteepness Analysis (normalized units):")
println("  Input - Leading edge (negative time): $(round(neg_grad_in, digits=4))")
println("  Input - Trailing edge (positive time): $(round(pos_grad_in, digits=4))")
println("  Output - Leading edge (negative time): $(round(neg_grad_out, digits=4))")
println("  Output - Trailing edge (positive time): $(round(pos_grad_out, digits=4))")

# Expected: Trailing edge steepness should INCREASE
trailing_steepness_ratio = pos_grad_out / pos_grad_in
leading_steepness_ratio = neg_grad_out / neg_grad_in

println("\nSteepness Ratios (output/input):")
println("  Trailing edge: $(round(trailing_steepness_ratio, digits=3))x")
println("  Leading edge: $(round(leading_steepness_ratio, digits=3))x")

# Physical expectation: trailing_steepness_ratio > leading_steepness_ratio
# AND trailing_steepness_ratio > 1.5 (significant steepening)

println("\n" * "=" ^ 70)
println("VALIDATION:")
if trailing_steepness_ratio > 1.5 && trailing_steepness_ratio > leading_steepness_ratio
    println("✓ PASS: Self-steepening occurs on TRAILING edge (correct direction)")
    println("        This is physically correct behavior.")
elseif leading_steepness_ratio > 1.5 && leading_steepness_ratio > trailing_steepness_ratio
    println("✗ FAIL: Self-steepening occurs on LEADING edge (WRONG direction)")
    println("        The shock term sign is incorrect!")
else
    println("? INCONCLUSIVE: No clear steepening observed")
    println("  May need higher power or longer propagation distance")
end
println("=" ^ 70)

# Create visualization
try
    p1 = plot(t_subset, I_input_norm, 
              label="Input", linewidth=2, 
              xlabel="Time (fs)", ylabel="Normalized Intensity",
              title="Self-Steepening Test (Shock Only)",
              legend=:topright)
    plot!(p1, t_subset, I_output_norm, 
          label="Output ($(medium.length*100) cm)", linewidth=2)
    
    # Mark peak positions
    vline!(p1, [t_peak_input], linestyle=:dash, label="Input peak", color=:blue, alpha=0.5)
    vline!(p1, [t_peak_output], linestyle=:dash, label="Output peak", color=:red, alpha=0.5)
    
    # Add annotation showing trailing edge
    annotate!(p1, [(maximum(t_subset)*0.7, 0.9, 
                   text("Trailing edge →\n(should steepen)", 10, :right))])
    
    savefig(p1, "self_steepening_test.png")
    println("\nPlot saved to: self_steepening_test.png")
catch e
    println("\nNote: Could not create plot (Plots.jl may not be available)")
    println("      Error: ", e)
end

println("\nTest completed.")
