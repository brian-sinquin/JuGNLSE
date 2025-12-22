using Test
using JuGNLSE

@testset "Physics: Solitons" begin
    # Parameters
    N = 1024
    T = 20e-12
    lambda0 = 1550e-9
    grid = create_grid(N, T, lambda0)

    T0 = 1e-12
    beta2 = -20e-27 # Anomalous
    gamma = 2.0
    P0 = calculate_soliton_power(beta2, gamma, T0)

    pulse = sech_pulse(grid, T0, P0; T0=true)

    L = 5.0 * (T0^2 / abs(beta2)) # 5 soliton periods (approx)
    medium = Medium(L, gamma, [beta2], 0.0, lambda0)

    for method in [:ERK4IP, :RK4IP, :SSFM]
        @testset "Method: $method" begin
            params = SimParams(; medium=medium, dz=L / 100, raman=false, shock=false)
            res = solve(pulse, params; method=method)

            final_At = res.At[:, end]

            # Check peak power remains constant
            @test peak_power(final_At) ≈ P0 rtol = 0.05

            # Check FWHM remains constant
            initial_fwhm = fwhm(abs2.(pulse.At), grid.t)
            final_fwhm = fwhm(abs2.(final_At), grid.t)
            @test final_fwhm ≈ initial_fwhm rtol = 0.05

            # Check shape (correlation)
            correlation =
                abs(sum(conj.(pulse.At) .* final_At)) /
                (sqrt(sum(abs2.(pulse.At))) * sqrt(sum(abs2.(final_At))))
            @test correlation ≈ 1.0 rtol = 0.01
        end
    end
end
