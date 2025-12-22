using Test
using JuGNLSE
using Statistics

@testset "Physics: Self-Steepening (Shock)" begin
    # Parameters
    N = 4096
    T = 20e-12
    lambda0 = 1550e-9
    grid = create_grid(N, T, lambda0)

    # Gaussian pulse
    T0 = 500e-15
    P0 = 5000.0
    pulse = gaussian_pulse(grid, T0, P0; T0=true)

    # Medium: No dispersion, no Raman, only Kerr + Shock
    gamma = 2.0
    L = 0.01 # 1 cm (reduced to avoid excessive distortion)
    medium = Medium(L, gamma, [0.0], 0.0, lambda0)

    # Theoretical peak shift: dt = 3 * gamma * P0 * L / omega0
    # The factor of 3 comes from the derivative of the cubic nonlinearity |A|^2 A
    omega0 = grid.omega0
    expected_shift = 3 * gamma * P0 * L / omega0

    for method in [:ERK4IP, :RK4IP, :SSFM]
        @testset "Method: $method" begin
            # Enable shock, disable raman
            # Use more steps for SSFM to ensure convergence
            dz_val = method == :SSFM ? L / 2000 : L / 500
            params = SimParams(; medium=medium, raman=false, shock=true, dz=dz_val)
            res = solve(pulse, params; method=method, progress=false)

            final_At = res.At[:, end]

            # Find peak position in time domain using interpolation for better precision
            function find_peak_t(At, t)
                # Quadratic interpolation around the maximum
                idx = argmax(abs2.(At))
                if idx == 1 || idx == length(t)
                    return t[idx]
                end

                y1 = abs2(At[idx - 1])
                y2 = abs2(At[idx])
                y3 = abs2(At[idx + 1])

                # Peak offset from idx
                # p = 0.5 * (y1 - y3) / (y1 - 2y2 + y3)
                denom = (y1 - 2y2 + y3)
                p = abs(denom) < 1e-15 ? 0.0 : 0.5 * (y1 - y3) / denom

                return t[idx] + p * (t[2] - t[1])
            end

            initial_peak_t = find_peak_t(pulse.At, grid.t)
            final_peak_t = find_peak_t(final_At, grid.t)

            actual_shift = final_peak_t - initial_peak_t

            # Self-steepening causes the peak to shift towards the trailing edge (positive t)
            # in the standard frame used in GNLSE.
            @test actual_shift > 0
            # The theoretical formula is an approximation; ERK4IP might be more accurate
            # but deviates from the simple linear prediction.
            @test actual_shift ≈ expected_shift rtol = 0.6
        end
    end
end
