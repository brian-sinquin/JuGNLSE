using Test
using JuGNLSE

@testset "Physics: Dispersion" begin
    # Parameters
    N = 1024
    T = 100e-12
    lambda0 = 1550e-9
    grid = create_grid(N, T, lambda0)

    T0 = 1e-12
    P0 = 1e-3 # Very low power to avoid nonlinearity
    pulse = gaussian_pulse(grid, T0, P0; T0=true)

    beta2 = 20e-27 # Normal dispersion
    L = 1000.0 # 1 km
    medium = Medium(L, 0.0, [beta2], 0.0, lambda0)

    LD = T0^2 / abs(beta2)
    expected_broadening = sqrt(1 + (L / LD)^2)

    for method in [:ERK4IP, :RK4IP, :SSFM]
        @testset "Method: $method" begin
            params = SimParams(; medium=medium, dz=10.0, raman=false, shock=false)
            res = solve(pulse, params; method=method)

            final_At = res.At[:, end]
            final_fwhm = fwhm(abs2.(final_At), grid.t)
            initial_fwhm = fwhm(abs2.(pulse.At), grid.t)

            broadening = final_fwhm / initial_fwhm
            @test broadening ≈ expected_broadening rtol = 1e-2
        end
    end
end
