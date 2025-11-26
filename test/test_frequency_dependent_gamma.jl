using Test
using JuGNLSE
using LinearAlgebra

@testset "Frequency-dependent gamma (M-GNLSE)" begin
    # Test parameters
    N = 2^10
    T_window = 10e-12  # 10 ps
    lambda0 = 835e-9   # 835 nm
    
    # Create grid
    grid = create_grid(N, T_window, lambda0)
    
    # Test 1: Medium construction with vector gamma (without scaling)
    @testset "Medium with vector gamma" begin
        gamma_vec = 0.11 .* ones(N)  # Constant gamma as vector
        medium = Medium(
            0.15,                          # length [m]
            gamma_vec,                     # vector gamma
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0
        )
        
        @test medium.gamma isa Vector
        @test length(medium.gamma) == N
        @test medium.scaling === nothing
    end
    
    # Test 2: Medium construction with scaling factor
    @testset "Medium with scaling factor" begin
        gamma_vec = 0.11 .* ones(N)
        scaling = ones(N)  # Unity scaling (no effect)
        
        medium = Medium(
            0.15,
            gamma_vec,
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0,
            scaling
        )
        
        @test medium.gamma isa Vector
        @test medium.scaling !== nothing
        @test length(medium.scaling) == N
    end
    
    # Test 3: Verify error handling
    @testset "Error handling" begin
        # Scaling with scalar gamma should not have a matching constructor
        @test_throws MethodError Medium(
            0.15,
            0.11,  # scalar
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0,
            ones(N)  # scaling provided
        )
        
        # Mismatched lengths should error
        @test_throws ArgumentError Medium(
            0.15,
            0.11 .* ones(N),
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0,
            ones(N-1)  # wrong length
        )
    end
    
    # Test 4: Comparison with scalar gamma (constant vector should give similar results)
    @testset "Scalar vs constant vector gamma" begin
        # Scalar gamma medium
        medium_scalar = Medium(
            0.15,
            0.11,
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0
        )
        
        # Vector gamma medium (constant value)
        gamma_vec = 0.11 .* ones(N)
        scaling = ones(N)  # Unity scaling
        medium_vector = Medium(
            0.15,
            gamma_vec,
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0,
            scaling
        )
        
        # Create identical pulses
        pulse_scalar = sech_pulse(grid, 50e-15, 10000.0, lambda0)
        pulse_vector = sech_pulse(grid, 50e-15, 10000.0, lambda0)
        
        # Setup parameters
        params_scalar = SimParams(
            medium=medium_scalar,
            N=N,
            n_saves=50,
            raman=false,
            shock=false
        )
        
        params_vector = SimParams(
            medium=medium_vector,
            N=N,
            n_saves=50,
            raman=false,
            shock=false
        )
        
        # Solve both
        results_scalar = solve(pulse_scalar, params_scalar, method=:rk4ip, progress=false)
        results_vector = solve(pulse_vector, params_vector, method=:rk4ip, progress=false)
        
        # Results should be very similar (not identical due to numerical differences)
        # Check final time-domain field
        At_scalar_final = results_scalar.At[:, end]
        At_vector_final = results_vector.At[:, end]
        
        # Relative error should be small
        rel_error = norm(At_scalar_final - At_vector_final) / norm(At_scalar_final)
        @test rel_error < 0.1  # 10% tolerance
        
        println("  Scalar vs vector gamma relative error: $(round(rel_error*100, digits=3))%")
    end
    
    # Test 5: Frequency-dependent gamma with varying values
    @testset "Frequency-varying gamma" begin
        # Create frequency-dependent gamma (higher at edges)
        omega0 = 2π * 3e8 / lambda0
        gamma_vec = 0.11 .* (1.0 .+ 0.1 .* (grid.omega ./ omega0).^2)
        
        # Create appropriate scaling (simplified, not physically accurate)
        scaling = ones(N)
        
        medium = Medium(
            0.15,
            gamma_vec,
            [0.0, 0.0, -11.83e-27, 8.13e-41],
            0.0,
            lambda0,
            scaling
        )
        
        pulse = sech_pulse(grid, 50e-15, 10000.0, lambda0)
        
        params = SimParams(
            medium=medium,
            N=N,
            n_saves=50,
            raman=false,
            shock=false
        )
        
        # Should solve without errors
        results = solve(pulse, params, method=:rk4ip, progress=false)
        
        # Check energy conservation (should be reasonable)
        energy_initial = sum(abs2.(results.At[:, 1])) * grid.dt
        energy_final = sum(abs2.(results.At[:, end])) * grid.dt
        energy_ratio = energy_final / energy_initial
        
        @test 0.5 < energy_ratio < 1.5  # Loose bounds, just checking it's not wildly wrong
        
        println("  Energy conservation ratio: $(round(energy_ratio, digits=3))")
    end
end
