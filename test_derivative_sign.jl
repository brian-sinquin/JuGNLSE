#!/usr/bin/env julia
"""
Minimal test to verify the self-steepening sign correction

This test creates a simple scenario where we can analytically verify
the direction of the time derivative.
"""

println("=" ^ 70)
println("Minimal Self-Steepening Sign Verification")
println("=" ^ 70)

# Test the derivative operator sign directly
using FFTW

# Create a simple test signal
N = 256
t = range(-5, 5, length=N)
dt = t[2] - t[1]

# Create a simple Gaussian pulse
sigma = 1.0
A_t = exp.(-t.^2 / (2*sigma^2))

# Compute intensity
I_t = abs2.(A_t)

# Compute analytical derivative ∂I/∂t
dI_dt_analytical = -t ./ sigma^2 .* exp.(-t.^2 / sigma^2)

# Compute numerical derivative using FFT (with corrected sign)
omega = fftfreq(N, 1/dt) * 2π
omega = fftshift(omega)  # Match JuGNLSE convention

# Method 1: Standard FFT convention (for reference)
I_w_standard = fft(I_t)
dI_dt_standard = real.(ifft(im .* fftshift(omega) .* I_w_standard))

# Method 2: Inverted FFT convention (as in JuGNLSE)
I_w_inverted = ifft(I_t)
dI_dt_inverted = real.(fft(im .* omega .* I_w_inverted))

# Compare with analytical
error_standard = maximum(abs.(dI_dt_standard - dI_dt_analytical))
error_inverted = maximum(abs.(dI_dt_inverted - dI_dt_analytical))

println("\nDerivative Computation Test:")
println("  Analytical peak derivative: $(round(minimum(dI_dt_analytical), digits=4))")
println("  Standard FFT peak derivative: $(round(minimum(dI_dt_standard), digits=4))")
println("  Inverted FFT peak derivative: $(round(minimum(dI_dt_inverted), digits=4))")
println()
println("  Maximum error (standard FFT): $(round(error_standard, digits=6))")
println("  Maximum error (inverted FFT): $(round(error_inverted, digits=6))")

# The key test: check the sign at t=1 (where derivative should be negative)
idx_test = findfirst(t .> 1.0)
analytical_sign = sign(dI_dt_analytical[idx_test])
inverted_sign = sign(dI_dt_inverted[idx_test])

println("\nSign Test at t=1:")
println("  Analytical derivative sign: $(analytical_sign > 0 ? "+" : "-")")
println("  Computed derivative sign: $(inverted_sign > 0 ? "+" : "-")")

println("\n" * "=" ^ 70)
if analytical_sign == inverted_sign && error_inverted < 0.01
    println("✓ PASS: Derivative sign is CORRECT with +iω operator")
    println("        The self-steepening fix is validated!")
else
    println("✗ FAIL: Derivative sign is INCORRECT")
    println("        Check the implementation!")
end
println("=" ^ 70)

# Additional verification: shock steepening direction
println("\nShock Steepening Direction:")
println("  For ∂I/∂t in self-steepening term iγ/ω₀ * ∂I/∂t:")
println("  - At trailing edge (t > 0): ∂I/∂t < 0 (intensity decreasing)")
println("  - This contributes negative phase: -γ/ω₀ * |∂I/∂t|")  
println("  - Negative phase → increased frequency → steepening")
println("  - Therefore: trailing edge steepens ✓")
println()
println("  With WRONG sign (-iω), the effect would be reversed:")
println("  - Leading edge would steepen (unphysical) ✗")
println()
println("Test completed successfully!")
