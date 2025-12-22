using Test
using JuGNLSE
using Statistics

@testset "Physics: Raman (SSFS)" begin
    # Parameters
    N = 4096
    T = 20e-12
    lambda0 = 1550e-9
    grid = create_grid(N, T, lambda0)

    # Fundamental soliton
    T0 = 50e-15
    beta2 = -20e-27 # Anomalous
    gamma = 2.0
    P0 = calculate_soliton_power(beta2, gamma, T0)
    pulse = sech_pulse(grid, T0, P0; T0=true)

    # Medium: Dispersion + Raman, no Shock
    L = 1.0 # 1 m (reduced to avoid wrap-around and instability)
    medium = Medium(L, gamma, [beta2], 0.0, lambda0)

    # Theoretical SSFS rate for Blow-Wood model
    # tau_R = fr * 2 * tau1^2 * tau2 / (tau1^2 + tau2^2)
    tau1 = 12.2e-15
    tau2 = 32.0e-15
    fr = 0.18
    tau_R = fr * (2 * tau1^2 * tau2) / (tau1^2 + tau2^2)

    # Delta_omega = -8 * tau_R * |beta2| * L / (15 * T0^4)
    expected_dw = -8 * tau_R * abs(beta2) * L / (15 * T0^4)

    for method in [:ERK4IP, :RK4IP, :SSFM]
        @testset "Method: $method" begin
            # Enable raman, disable shock
            # Use smaller dz for SSFM stability
            dz_val = method == :SSFM ? L / 2000 : L / 500
            params = SimParams(; medium=medium, raman=true, shock=false, dz=dz_val)
            res = solve(pulse, params; method=method, progress=false)

            final_Aw = res.Aw[:, end]

            # Calculate spectral center of mass shift
            # omega is detuning from omega0
            function spectral_cm(Aw, omega)
                spec = abs2.(Aw)
                return sum(omega .* spec) / sum(spec)
            end

            initial_cm = spectral_cm(pulse.Aw, grid.omega)
            final_cm = spectral_cm(final_Aw, grid.omega)

            actual_dw = final_cm - initial_cm

            # Raman causes a red-shift (negative frequency shift)
            @test actual_dw < 0
            @test ≈(actual_dw, expected_dw, rtol=0.3)
        end
    end
end
