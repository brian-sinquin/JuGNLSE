using JuGNLSE

# 1. Constants
lambda0 = 835e-9
gamma0 = 0.11

# 2. Test ZDependentGamma
println("--- Testing ZDependentGamma ---")
L = 0.15
taper_func = z -> gamma0 * (1 - 0.5 * z/L)
g_z = ZDependentGamma(taper_func)
println("Gamma(z=0) = ", gamma(g_z, lambda0, 0.0), " (Expected: 0.11)")
println("Gamma(z=L) = ", gamma(g_z, lambda0, L), " (Expected: 0.055)")

# 3. Test WavelengthDependentGamma
println("\n--- Testing WavelengthDependentGamma ---")
# Gamma(λ) = gamma0 * (lambda / lambda0)
lambda_func = λ -> gamma0 * (λ / lambda0)
g_l = WavelengthDependentGamma(lambda_func)
println("Gamma(λ=λ0) = ", gamma(g_l, lambda0, 0.0), " (Expected: 0.11)")
println("Gamma(λ=2*λ0) = ", gamma(g_l, 2*lambda0, 0.0), " (Expected: 0.22)")
