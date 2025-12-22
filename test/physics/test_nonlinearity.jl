using Test
using JuGNLSE

@testset "Physics: Nonlinearity (SPM)" begin
    # Parameters
    N = 1024
    T = 100e-12
    lambda0 = 1550e-9
    grid = create_grid(N, T, lambda0)

    T0 = 10e-12
    P0 = 1.0
    pulse = gaussian_pulse(grid, T0, P0; T0=true)

    gamma = 2.0
    L = 0.5
    # No dispersion
    medium = Medium(L, gamma, [0.0], 0.0, lambda0)

    expected_phase_max = gamma * P0 * L

    for method in [:ERK4IP, :RK4IP, :SSFM]
        @testset "Method: $method" begin
            params = SimParams(; medium=medium, dz=0.01, raman=false, shock=false)
            res = solve(pulse, params; method=method)

            final_At = res.At[:, end]
            # Phase is angle(At)
            # We need to unwrap or just check the peak
            phase = angle.(final_At)
            # The pulse is centered at t=0, which is at index N/2 + 1
            peak_idx = argmax(abs2.(pulse.At))
            actual_phase_max = phase[peak_idx]

            # Note: angle returns [-pi, pi]. If expected > pi, we need to be careful.
            # Here expected = 2.0 * 1.0 * 0.5 = 1.0 rad, which is fine.
            @test actual_phase_max ≈ expected_phase_max rtol = 1e-2
        end
    end
end
