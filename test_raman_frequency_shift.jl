"""
Test to validate Raman-induced frequency shift direction

Physical expectation: Fundamental soliton with Raman should experience
continuous red-shift (shift to longer wavelengths, lower frequencies).
This is the soliton self-frequency shift (SSFS) effect.

This test creates an N=1 soliton and propagates with Raman enabled
to verify the direction of frequency shift.
"""

using JuGNLSE

println("=" ^ 70)
println("Raman-Induced Frequency Shift Test")
println("=" ^ 70)

# Create grid
N = 2^12
time_window = 10e-12  # 10 ps
center_wavelength = 835e-9  # 835 nm
grid = create_grid(N, time_window, center_wavelength)

# Create medium with anomalous dispersion (β₂ < 0) for soliton
# Using typical PCF parameters
medium = Medium(
    0.10,                    # 10 cm fiber
    0.11,                    # γ = 0.11 W⁻¹m⁻¹ (typical PCF)
    [-11.83e-27, 8.1e-41],  # β₂, β₃ (anomalous dispersion)
    0.0,                     # No loss
    center_wavelength
)

# Calculate N=1 fundamental soliton power
beta2 = medium.betas[1]
T0 = 50e-15  # 50 fs FWHM / 1.76 ≈ 28.4 fs
P0 = abs(beta2) / (medium.gamma * T0^2)

println("\nSoliton Parameters:")
println("  β₂ = $(beta2*1e27) ps²/km")
println("  T₀ = $(T0*1e15) fs")
println("  P₀ = $(round(P0/1e3, digits=2)) kW (N=1 soliton)")
println("  Fiber length: $(medium.length*100) cm")
println()

# Create fundamental soliton pulse
pulse = sech_pulse(grid, T0, P0, center_wavelength, T0=true)

# Configure simulation: Raman only (no self-steepening for clearer test)
params = SimParams(
    medium = medium,
    N = N,
    n_saves = 100,
    raman = true,          # Enable Raman
    shock = false,         # Disable self-steepening for clearer signal
    raman_model = BlowWood(),
    reltol = 1e-7,
    abstol = 1e-9
)

println("Propagating with Raman scattering enabled...")
z, At, Aw = solve(pulse, params, method=:rk4ip)

# Analysis: Measure spectral shift
function find_spectral_peak(Aw_slice, omega)
    spectrum = abs2.(Aw_slice)
    idx_peak = argmax(spectrum)
    return omega[idx_peak]
end

# Calculate peak frequency at input and output
omega_input = find_spectral_peak(Aw[:, 1], grid.omega)
omega_output = find_spectral_peak(Aw[:, end], grid.omega)

# Convert to wavelength
c = 3e8  # m/s
omega0 = 2π * c / center_wavelength
lambda_input = 2π * c / (omega0 + omega_input)
lambda_output = 2π * c / (omega0 + omega_output)

# Calculate shifts
delta_omega = omega_output - omega_input
delta_lambda = lambda_output - lambda_input

println("\nResults:")
println("  Input peak wavelength: $(round(lambda_input*1e9, digits=2)) nm")
println("  Output peak wavelength: $(round(lambda_output*1e9, digits=2)) nm")
println("  Wavelength shift: $(round(delta_lambda*1e9, digits=2)) nm")
println("  Frequency shift: $(round(delta_omega/(2π)*1e-12, digits=3)) THz")

# Check if shift is to red (positive wavelength shift, negative frequency shift)
println("\n" * "=" ^ 70)
println("VALIDATION:")
if delta_lambda > 0 && delta_omega < 0
    println("✓ PASS: Raman causes RED-SHIFT (shift to longer wavelengths)")
    println("        This is the correct Stokes shift direction.")
    println("        Magnitude: $(round(delta_lambda*1e9, digits=2)) nm shift")
elseif delta_lambda < 0 && delta_omega > 0
    println("✗ FAIL: Raman causes BLUE-SHIFT (WRONG direction)")
    println("        The Raman response or convolution has incorrect sign!")
else
    println("? INCONCLUSIVE: No significant frequency shift observed")
    println("  May need longer fiber or different parameters")
end
println("=" ^ 70)

# Additional check: Measure shift rate (should be approximately constant)
n_check = min(10, size(Aw, 2))
z_check = z[1:div(size(Aw, 2), n_check):end]
omega_check = [find_spectral_peak(Aw[:, i], grid.omega) 
               for i in 1:div(size(Aw, 2), n_check):size(Aw, 2)]

# Calculate average shift rate
if length(z_check) > 2
    shift_rate = (omega_check[end] - omega_check[1]) / (z_check[end] - z_check[1])
    println("\nShift rate: $(round(shift_rate/(2π)*1e-12*1e3, digits=3)) THz/m")
    
    # For reference: SSFS typically gives shift rate proportional to gamma*P0/T0^4
    # Positive rate means red-shift (negative frequency shift)
    if shift_rate < 0
        println("✓ Shift rate is NEGATIVE (red-shift with propagation)")
    else
        println("✗ Shift rate is POSITIVE (blue-shift, WRONG)")
    end
end

println("\nTest completed.")
