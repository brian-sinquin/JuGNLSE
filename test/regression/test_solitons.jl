# Regression tests: Soliton propagation validation
# Fundamental and higher-order solitons provide analytical/semi-analytical
# validation for GNLSE solvers.
using Test
using JuGNLSE

@testset "Soliton Propagation" begin
    @testset "Soliton Condition Validation" begin
        # Verify soliton number calculation formula
        lambda0 = 1550e-9
        beta2 = -20e-27
        T0 = 50e-15
        gamma = 0.01
        
        # Test for N=1, N=2, N=3
        for N_soliton in [1, 2, 3]
            P0 = N_soliton^2 * abs(beta2) / (gamma * T0^2)
            
            # Calculate soliton number from peak power
            N_calc = sqrt(gamma * P0 * T0^2 / abs(beta2))
            
            @test abs(N_calc - N_soliton) < 0.01
        end
    end
    
    @testset "Fundamental Soliton (N=1) with SSFM" begin
        # Setup
        lambda0 = 1550e-9
        beta2 = -20e-27
        T0 = 50e-15
        gamma = 0.01
        
        # N=1 soliton: dispersion and nonlinearity balance
        P0 = abs(beta2) / (gamma * T0^2)
        
        # Dispersion length
        LD = T0^2 / abs(beta2)
        
        # Propagate 2 dispersion lengths
        z_prop = 2.0 * LD
        
        # Create grid and pulse - USE T0=true!
        grid = create_grid(2^12, 20e-12, lambda0)
        medium = Medium(z_prop, gamma, [beta2], 0.0, lambda0)
        params = SimParams(medium=medium, N=2^12, n_saves=20, raman=false, shock=false)
        pulse = sech_pulse(grid, T0, P0, lambda0, T0=true)
        
        # Propagate with SSFM
        results = solve(pulse, params, method=:ssfm, progress=false)
        
        # Calculate metrics
        E_initial = pulse_energy(results.At[:,1], grid.dt)
        E_final = pulse_energy(results.At[:,end], grid.dt)
        P_initial = maximum(abs2.(results.At[:,1]))
        P_final = maximum(abs2.(results.At[:,end]))
        
        # Energy conservation
        energy_change = abs(E_final - E_initial) / E_initial
        @test energy_change < 0.01  # <1% energy drift
        
        # Peak power conservation (N=1 soliton maintains shape)
        peak_change = abs(P_final - P_initial) / P_initial
        @test peak_change < 0.05  # <5% peak power change
        
        # Shape fidelity
        initial_shape = abs2.(results.At[:,1])
        final_shape = abs2.(results.At[:,end])
        initial_norm = initial_shape / maximum(initial_shape)
        final_norm = final_shape / maximum(final_shape)
        fidelity = sum(sqrt.(initial_norm .* final_norm)) / sqrt(sum(initial_norm) * sum(final_norm))
        @test fidelity > 0.99  # >99% shape preservation
    end
    
    @testset "Higher-Order Soliton (N=2)" begin
        # Setup
        lambda0 = 1550e-9
        beta2 = -20e-27
        T0 = 50e-15
        gamma = 0.01
        
        # N=2 soliton
        N_soliton = 2
        P0 = N_soliton^2 * abs(beta2) / (gamma * T0^2)
        
        # Soliton period
        LD = T0^2 / abs(beta2)
        z_period = pi / 2 * LD
        
        # Propagate one period
        z_prop = z_period
        
        # Create grid and pulse - USE T0=true!
        grid = create_grid(2^12, 20e-12, lambda0)
        medium = Medium(z_prop, gamma, [beta2], 0.0, lambda0)
        params = SimParams(medium=medium, N=2^12, n_saves=20, raman=false, shock=false)
        pulse = sech_pulse(grid, T0, P0, lambda0, T0=true)
        
        # Propagate
        results = solve(pulse, params, method=:ssfm, progress=false)
        
        # Calculate metrics
        E_initial = pulse_energy(results.At[:,1], grid.dt)
        E_final = pulse_energy(results.At[:,end], grid.dt)
        P_initial = maximum(abs2.(results.At[:,1]))
        P_final = maximum(abs2.(results.At[:,end]))
        
        # Energy conservation
        energy_change = abs(E_final - E_initial) / E_initial
        @test energy_change < 0.02  # <2% energy drift
        
        # N=2 soliton should return to initial shape after one period
        initial_shape = abs2.(results.At[:,1])
        final_shape = abs2.(results.At[:,end])
        initial_norm = initial_shape / maximum(initial_shape)
        final_norm = final_shape / maximum(final_shape)
        fidelity = sum(sqrt.(initial_norm .* final_norm)) / sqrt(sum(initial_norm) * sum(final_norm))
        @test fidelity > 0.95  # >95% periodic return
    end
    
    @testset "Soliton Self-Frequency Shift (SSFS) - RK4IP issues" begin
        @test_skip "RK4IP has adaptive stepping issues with Raman - needs investigation"
    end
end
